"""An independent golden baseline for the thirteen pre-existing funnel columns.

Why this module exists
----------------------
Section 14 of the Phase 5.3 contract says what must be identical across a
regeneration and what is expected to change: the thirteen probe columns must
be identical in VALUE, while the whole-file hash must change, because seven
columns were added. The Phase 5.3-B publish gate enforced exactly that - and
then ``os.replace`` overwrote the artifact it had just compared against. The
comparison was real but it left no evidence behind, so "the thirteen columns
never moved" rested on a check that had already consumed its own baseline and
that no test exercised.

This module makes the baseline a durable artifact of its own. It is small
enough to live in the repository beside the tests, it is readable, and it
survives every regeneration that follows.

What a digest here actually proves
----------------------------------
Each column is hashed over its exact IEEE-754 / two's-complement bytes in row
order, so the digest is an element-wise assertion over all 253,429 values of
that column, not a summary of them. One flipped value in one row changes the
digest. Row ORDER is covered twice: implicitly, because the bytes are laid out
in row order, and explicitly by ``row_order_digest`` over ``edge_index``, so a
permuted artifact with identical multiset contents still fails and fails with
a message that says which of the two went wrong.

The summary block beside each digest is not the check. It is there so that a
failure says "``pooled_a`` sum moved by 12" instead of only "digest differs",
because a bare digest mismatch gives an investigator nowhere to start.

Independence
------------
A baseline derived by the same code path that wrote the artifact would prove
only that the path is deterministic. Two derivations are therefore compared:

1. :func:`digest_frame` reading the published parquet directly, and
2. the same function over a funnel built through the **Phase 5.2 code path** -
   ``build_funnel`` with no ``production_statistics_config``, which runs the
   single probe walk and emits thirteen columns exactly as it did before
   schema /2 existed.

Agreement between those two is the finding worth having: it says the seven
added columns and the second walk that produces them perturbed nothing in the
thirteen, measured against a derivation that does not know they exist.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

#: Version of the baseline document itself, so a future format change to this
#: file is a loud mismatch rather than a silent misread.
GOLDEN_SCHEMA = "obsidianchain.funnel_golden/1"

#: Domain separator, for the same reason ``run_fingerprint`` carries one: a
#: digest here must never be confusable with a plain file hash.
DOMAIN = "obsidianchain.funnel_golden.column/1"

#: The thirteen columns that predate schema /2, and the dtype each is hashed
#: under. Fixing the dtype is what makes a digest reproducible: reading the
#: same parquet with a different pandas could otherwise widen an int32 to an
#: int64 and change every byte without changing a single value.
PROBE_COLUMN_DTYPES: dict[str, str] = {
    "edge_index": "int64",
    "node_a": "int64",
    "node_b": "int64",
    "pooled_a": "int64",
    "pooled_b": "int64",
    "min_pooled": "int64",
    "size_a": "int64",
    "size_b": "int64",
    "min_size": "int64",
    "dof": "int64",
    "chi2": "float64",
    "p_value": "float64",
    "effect": "float64",
}

#: Order is part of the contract, as it is for the fingerprint's field order.
PROBE_COLUMNS: tuple[str, ...] = tuple(PROBE_COLUMN_DTYPES)

#: Columns digested at a fixed RELATIVE precision instead of bit-exactly,
#: and the only column that needs it.
#:
#: Measured, not assumed. Re-deriving the thirteen columns from the same
#: inputs and comparing against the published artifact:
#:
#: * on **amd64**, the architecture that generated the artifact, all thirteen
#:   columns are bit-identical - 0 of 253,429 rows differ;
#: * on **arm64**, twelve columns are bit-identical and ``p_value`` differs in
#:   608 of its 1,515 finite rows, by at most 5.55e-16 absolute (1-3 ULPs).
#:
#: The cause is ``scipy.stats.chi2.sf`` - an incomplete gamma whose last bits
#: follow the platform's libm. ``chi2`` and ``effect`` are immune only because
#: section 21 rounds them to six decimals, which absorbs the noise;
#: ``p_value`` is deliberately NOT in ``ROUNDED_COLUMNS``, because rounding a
#: p-value of 4.45e-37 to six decimals would destroy it.
#:
#: So a bit-exact digest on this one column would assert the architecture
#: rather than the values, fail on every arm64 dev machine, and be deleted by
#: the third engineer who hits it - which is the same failure as a test that
#: silently skips. Twelve significant digits is four orders tighter than any
#: change worth catching (a real one moves a p-value by orders of magnitude,
#: not by one ULP) and is stable across libm implementations.
TOLERANT_COLUMNS: tuple[str, ...] = ("p_value",)

#: Significant digits - not decimal places. ``%.11e`` keeps twelve
#: significant digits at every magnitude, so a p-value of 4.45e-37 is
#: compared just as strictly as one of 0.078.
SIGNIFICANT_DIGITS = 12


class GoldenMismatchError(AssertionError):
    """Raised when a frame does not match the golden baseline."""


def _canonical_bytes(series: pd.Series, dtype: str, *, tolerant: bool) -> bytes:
    """The exact bytes a column is hashed over.

    Bit-exact for twelve of the thirteen columns: NaN is left as NaN rather
    than substituted, because ``numpy``'s quiet NaN has one bit pattern and
    every NaN here comes from ``np.nan``, while a sentinel would make a real
    value that happened to equal it indistinguishable from a missing one.

    For a ``TOLERANT_COLUMNS`` member the encoding is the decimal string at
    :data:`SIGNIFICANT_DIGITS`, which is still one token per row in row order
    - an element-wise assertion over every value, just at twelve significant
    digits instead of fifty-three bits. See ``TOLERANT_COLUMNS`` for why that
    one column cannot be held to its bits.
    """
    array = np.ascontiguousarray(series.to_numpy(dtype=dtype))
    if not tolerant:
        return array.tobytes()
    precision = SIGNIFICANT_DIGITS - 1
    return "\n".join(
        "nan" if np.isnan(value) else f"{value:.{precision}e}"
        for value in array
    ).encode()


def column_digest(series: pd.Series, dtype: str, *, tolerant: bool = False) -> str:
    digest = hashlib.sha256()
    mode = f"sig{SIGNIFICANT_DIGITS}" if tolerant else "bits"
    digest.update(f"{DOMAIN}\n{dtype}\n{mode}\n{len(series)}\n".encode())
    digest.update(_canonical_bytes(series, dtype, tolerant=tolerant))
    return digest.hexdigest()


def _summary(series: pd.Series, dtype: str) -> dict:
    """Human-readable shape of a column, for diagnosing a digest mismatch."""
    array = series.to_numpy(dtype=dtype)
    finite = array[np.isfinite(array)] if dtype == "float64" else array
    return {
        "n_nan": int(np.isnan(array).sum()) if dtype == "float64" else 0,
        "min": None if finite.size == 0 else float(finite.min()),
        "max": None if finite.size == 0 else float(finite.max()),
        "sum": None if finite.size == 0 else float(finite.sum()),
    }


def digest_frame(frame: pd.DataFrame) -> dict:
    """Digest the thirteen probe columns of ``frame``.

    Selects by NAME, never by position. Phase 4 prepends two provenance
    columns and schema /2 appends seven more, so a positional slice would be
    correct only for one particular layout - and reporting a mismatch because
    the columns MOVED, when every value is intact, is exactly the kind of
    false alarm that gets a guard disabled.
    """
    missing = [c for c in PROBE_COLUMNS if c not in frame.columns]
    if missing:
        raise GoldenMismatchError(
            f"the frame is missing probe columns {missing}; a baseline cannot "
            f"be taken over columns that are not there"
        )
    columns = {
        name: {
            "dtype": dtype,
            "comparison": (
                f"significant-digits/{SIGNIFICANT_DIGITS}"
                if name in TOLERANT_COLUMNS else "bit-exact"
            ),
            "digest": column_digest(
                frame[name], dtype, tolerant=name in TOLERANT_COLUMNS
            ),
            "summary": _summary(frame[name], dtype),
        }
        for name, dtype in PROBE_COLUMN_DTYPES.items()
    }
    return {
        "schema": GOLDEN_SCHEMA,
        "n_rows": int(len(frame)),
        "column_order": list(PROBE_COLUMNS),
        # Row order, asserted separately from the values. edge_index is the
        # row's identity, so its digest pins the sequence itself.
        "row_order_digest": column_digest(frame["edge_index"], "int64"),
        "columns": columns,
    }


def write_baseline(frame: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(digest_frame(frame), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def read_baseline(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise GoldenMismatchError(
            f"{path} not found. The golden baseline is the only record that "
            f"the thirteen pre-existing columns never moved; it is not "
            f"regenerable from an artifact that has already been replaced."
        )
    baseline = json.loads(path.read_text(encoding="utf-8"))
    if baseline.get("schema") != GOLDEN_SCHEMA:
        raise GoldenMismatchError(
            f"{path} declares schema {baseline.get('schema')!r}; "
            f"{GOLDEN_SCHEMA} expected"
        )
    return baseline


def compare(frame: pd.DataFrame, baseline: dict) -> list[str]:
    """Return one message per difference. Empty means identical.

    A list rather than a raised exception so a caller can report every column
    that moved at once. One-at-a-time failure on thirteen columns turns a
    single regeneration bug into thirteen debugging rounds.
    """
    problems: list[str] = []
    fresh = digest_frame(frame)

    if fresh["n_rows"] != baseline["n_rows"]:
        problems.append(
            f"row count {fresh['n_rows']:,} != baseline "
            f"{baseline['n_rows']:,}"
        )
    if fresh["column_order"] != baseline["column_order"]:
        problems.append(
            f"probe column order {fresh['column_order']} != baseline "
            f"{baseline['column_order']}"
        )
    if fresh["row_order_digest"] != baseline["row_order_digest"]:
        problems.append(
            "ROW ORDER changed: the edge_index sequence does not match the "
            "baseline, so row i of the new artifact is not row i of the old "
            "one even where the values agree"
        )

    for name in PROBE_COLUMNS:
        want = baseline["columns"].get(name)
        if want is None:
            problems.append(f"{name}: absent from the baseline")
            continue
        got = fresh["columns"][name]
        if got["digest"] == want["digest"]:
            continue
        moved = [
            f"{key} {want['summary'][key]} -> {got['summary'][key]}"
            for key in ("n_nan", "min", "max", "sum")
            if want["summary"][key] != got["summary"][key]
        ]
        problems.append(
            f"{name}: VALUES CHANGED ({got['comparison']}; digest "
            f"{want['digest'][:16]} -> {got['digest'][:16]})"
            + (f"; {', '.join(moved)}" if moved else
               "; summary statistics identical, so the change is a "
               "rearrangement or an offsetting pair of edits")
        )
    return problems


def compare_values(fresh: pd.DataFrame, old: pd.DataFrame) -> list[str]:
    """Element-wise comparison of two frames' thirteen probe columns.

    The digest comparison above is the durable one - it survives the artifact
    it was taken from. This is the direct one, for the moment when both the
    old and the new artifact are in hand: it reports how MANY rows differ and
    by how much, which a digest cannot.

    ``p_value`` is compared at a relative tolerance for the reason given in
    ``TOLERANT_COLUMNS``; the other twelve must be exactly equal, NaN
    positions included.
    """
    problems: list[str] = []
    if len(fresh) != len(old):
        return [f"row count {len(fresh):,} != {len(old):,}; cannot align rows"]

    for name, dtype in PROBE_COLUMN_DTYPES.items():
        a = fresh[name].to_numpy(dtype=dtype)
        b = old[name].to_numpy(dtype=dtype)
        if dtype == "float64":
            nan_a, nan_b = np.isnan(a), np.isnan(b)
            if not np.array_equal(nan_a, nan_b):
                problems.append(
                    f"{name}: NaN positions differ in "
                    f"{int((nan_a != nan_b).sum()):,} rows"
                )
                continue
            fa, fb = a[~nan_a], b[~nan_b]
            if name in TOLERANT_COLUMNS:
                tolerance = 10.0 ** -(SIGNIFICANT_DIGITS - 1)
                close = np.isclose(fa, fb, rtol=tolerance, atol=0.0)
                if not close.all():
                    worst = np.abs((fa - fb) / np.where(fb == 0, 1, fb)).max()
                    problems.append(
                        f"{name}: {int((~close).sum()):,} rows exceed rtol "
                        f"{tolerance:.0e}; worst relative deviation {worst:.3e}"
                    )
                continue
            if not np.array_equal(fa, fb):
                differing = int((fa != fb).sum())
                problems.append(
                    f"{name}: {differing:,} finite rows differ; max absolute "
                    f"deviation {float(np.abs(fa - fb).max()):.3e}"
                )
        elif not np.array_equal(a, b):
            problems.append(f"{name}: {int((a != b).sum()):,} rows differ")
    return problems
