"""Render the demonstration payload - terminal and a self-contained HTML page.

The marking is data, not decoration
-----------------------------------
Both renderers call :func:`obsidianchain.demo.api.assert_demo_flagged` before
producing a single character, and both draw the banner, the provenance string
and the per-scenario badge **from the payload's own fields**. So the marking
cannot be lost by editing a template: an unflagged payload does not render at
all, and a flagged one carries its warning into every card on the page.

The page states both things at once, in the same visual weight: what the
mechanism does, and that the frozen Elliptic++ dataset does not trigger it.
Putting the second in a footnote would make the first a lie.

Offline
-------
No stylesheet, script or font is fetched. The whole page is one file with
inline CSS, which is the only thing that works in an air-gapped container and
also happens to be the only thing that works from a USB stick on a judge's
laptop.
"""

from __future__ import annotations

import html
from pathlib import Path

from obsidianchain.demo.api import assert_demo_flagged

#: Colour per outcome. Deliberately not a red/green pass-fail scale - none of
#: these five outcomes is a failure, and four of the five end in a merge.
OUTCOME_STYLE = {
    "CANDIDATE": ("#4a5568", "#e2e8f0"),
    "MERGED": ("#2c5282", "#bee3f8"),
    "BLOCKED": ("#822727", "#fed7d7"),
    "CONTESTED": ("#744210", "#fefcbf"),
    "ABSTAINED": ("#2d3748", "#e2e8f0"),
}

OUTCOME_GLOSS = {
    "CANDIDATE": "co-spend proposed it; the network layer was never consulted",
    "MERGED": "the rule looked and found no separation",
    "BLOCKED": "the cannot-link fired and the merge was refused",
    "CONTESTED": "contradiction found after the merge; recorded, not resolved",
    "ABSTAINED": "observations existed and none of them were usable",
}


# ---- terminal ----------------------------------------------------------

_W = 78


def format_terminal(envelope: dict) -> str:
    """The demonstration as a fixed-width report."""
    assert_demo_flagged(envelope)
    out: list[str] = []
    add = out.append
    banner = envelope["banner"]

    add("=" * _W)
    add(f"  *** {banner} ***")
    add("=" * _W)
    add("  OBSIDIANCHAIN - CONSTRAINT MECHANISM, FIVE DEMONSTRATION SCENARIOS")
    add("")
    add(f"  namespace   {envelope['namespace']}")
    add(f"  seed        {envelope['seed']}   (reproducible; rebuild is "
        f"byte-identical)")
    add(f"  provenance  {envelope['provenance']}")
    add("")
    for line in _wrap(envelope["statements"]["mechanism"], _W - 6):
        add(f"  {line}")
    add("")
    add("-- what this does NOT show " + "-" * (_W - 27))
    for line in _wrap(envelope["statements"]["frozen_dataset"], _W - 6):
        add(f"  {line}")
    add("")

    fixture, rule = envelope["fixture"], envelope["rule"]
    add("-- fixture " + "-" * (_W - 11))
    add(f"  addresses                     {fixture['addresses']:>12,}")
    add(f"  co-spend edges                {fixture['cospend_edges']:>12,}")
    add(f"  chain transactions            {fixture['chain_transactions']:>12,}")
    add(f"  announced                     {fixture['announced_transactions']:>12,}")
    add(f"  never announced               {fixture['unannounced_transactions']:>12,}"
        f"   scenario A")
    add(f"  announcement records          {fixture['announcement_records']:>12,}")
    add(f"  usable after classification   {fixture['usable_transactions']:>12,}")
    add(f"  discarded NO_EVIDENCE         {fixture['discarded_no_evidence']:>12,}"
        f"   never reach a constraint")
    add(f"  sigma                         {fixture['sigma']:>12.4f}"
        f"   {fixture['sigma_source']}")
    add(f"  observations sha256           {fixture['dataset_sha256'][:16]}...")
    add("")

    add("-- the rule, unchanged from production " + "-" * (_W - 39))
    add(f"  min pooled per side           {rule['min_pooled_observations']:>12,}")
    add(f"  alpha                         {rule['alpha']:>12.1e}")
    add(f"  min effect                    {rule['min_effect']:>12.3f}")
    for line in _wrap(rule["note"], _W - 6):
        add(f"  {line}")
    add("")

    add("-- scenarios " + "-" * (_W - 13))
    add(f"  {'':<3}{'outcome':<11}{'verdict':<15}{'pooled':<14}{'p':<11}effect")
    for scenario in envelope["scenarios"]:
        headline = _headline_numbers(scenario)
        add(f"  {scenario['key']:<3}{scenario['outcome']:<11}"
            f"{scenario['verdict']:<15}{headline['pooled']:<14}"
            f"{headline['p']:<11}{headline['effect']}")
    add("")

    for scenario in envelope["scenarios"]:
        add("-" * _W)
        add(f"  {scenario['key']}  {scenario['title'].upper()}")
        add(f"     [{envelope['provenance']}]  ->  {scenario['outcome']}")
        add("")
        for label, text in (
            ("situation", scenario["situation"]),
            ("chain", scenario["chain_story"]),
            ("network", scenario["network_story"]),
        ):
            lines = _wrap(text, _W - 16)
            add(f"     {label:<12}{lines[0]}")
            for line in lines[1:]:
                add(f"     {'':<12}{line}")
        add("")
        for decision in scenario["decisions"][:4]:
            add(f"     {decision['address_a']} + {decision['address_b']}")
            add(f"       {decision['verdict']:<15}"
                f"pooled {decision['pooled_a']}/{decision['pooled_b']}   "
                f"p={decision['p_value']:.2e}  effect={decision['effect']:.4f}"
                f"   {'merged' if decision['merged'] else 'REFUSED'}")
        if len(scenario["decisions"]) > 4:
            remaining = len(scenario["decisions"]) - 4
            add(f"     ... {remaining} further decision(s), all "
                f"{scenario['decisions'][-1]['verdict']}")
        contradiction = scenario.get("contradiction")
        if contradiction:
            add("")
            add(f"     CONTRADICTION  one member against the other "
                f"{contradiction['component_size'] - 1} in its component")
            add(f"       {contradiction['verdict']:<15}"
                f"pooled {contradiction['pooled_member']}/"
                f"{contradiction['pooled_rest']}   "
                f"p={contradiction['p_value']:.2e}  "
                f"effect={contradiction['effect']:.4f}")
            add(f"       found by  {contradiction['found_by']}")
            add(f"       action    {contradiction['resolution']}")
        add("")
        for index, line in enumerate(_wrap(scenario["reads"], _W - 8)):
            add(f"     > {line}" if index == 0 else f"       {line}")
        add("")

    totals = envelope["totals"]
    add("=" * _W)
    add(f"  clusters chain-only {totals['chain_only_clusters']}"
        f"  ->  fused {totals['fused_clusters']}"
        f"   merges blocked {totals['merges_blocked']}"
        f"   contested {totals['components_contested']}")
    add(f"  decisions evaluated {totals['decisions_evaluated']}"
        f"   abstained {totals['decisions_abstained']}")
    add(f"  every scenario as described: "
        f"{'yes' if totals['all_scenarios_as_described'] else 'NO'}")
    add("")
    add(f"  *** {banner} ***")
    add("=" * _W)
    return "\n".join(out)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]


def _headline_numbers(scenario: dict) -> dict:
    """The one decision a reader should look at, per scenario."""
    source = scenario.get("contradiction")
    if source:
        return {
            "pooled": f"{source['pooled_member']}/{source['pooled_rest']}",
            "p": f"{source['p_value']:.1e}",
            "effect": f"{source['effect']:.4f}",
        }
    decisions = scenario["decisions"]
    if not decisions:
        return {"pooled": "-", "p": "-", "effect": "-"}
    decision = next((d for d in decisions if not d["merged"]), decisions[0])
    return {
        "pooled": f"{decision['pooled_a']}/{decision['pooled_b']}",
        "p": f"{decision['p_value']:.1e}",
        "effect": f"{decision['effect']:.4f}",
    }


# ---- HTML --------------------------------------------------------------

_CSS = """
:root {
  --ink: #1a202c; --muted: #4a5568; --line: #cbd5e0; --page: #f7fafc;
  --card: #ffffff; --warn-bg: #742a2a; --warn-ink: #fffaf0;
  --stripe-a: #7b341e; --stripe-b: #9c4221;
}
@media (prefers-color-scheme: dark) {
  :root {
    --ink: #e2e8f0; --muted: #a0aec0; --line: #2d3748; --page: #12161d;
    --card: #1a202c; --warn-bg: #822727; --warn-ink: #fffaf0;
    --stripe-a: #742a2a; --stripe-b: #9c4221;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
}
.hazard {
  position: sticky; top: 0; z-index: 20;
  background: repeating-linear-gradient(45deg,
    var(--stripe-a) 0 18px, var(--stripe-b) 18px 36px);
  color: var(--warn-ink); text-align: center;
  padding: 11px 16px; font-weight: 800; letter-spacing: .11em;
  font-size: 13px; text-transform: uppercase;
  border-bottom: 3px solid var(--warn-bg);
}
.hazard small {
  display: block; font-weight: 600; letter-spacing: .04em;
  text-transform: none; opacity: .92; margin-top: 3px; font-size: 12px;
}
.wrap { max-width: 1000px; margin: 0 auto; padding: 26px 20px 60px; }
h1 { font-size: 25px; margin: 12px 0 4px; letter-spacing: -.01em; }
h2 { font-size: 15px; text-transform: uppercase; letter-spacing: .09em;
     color: var(--muted); margin: 34px 0 12px; }
.sub { color: var(--muted); margin: 0 0 22px; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 760px) { .two { grid-template-columns: 1fr; } }
.panel { background: var(--card); border: 1px solid var(--line);
         border-radius: 9px; padding: 15px 17px; }
.panel h3 { margin: 0 0 7px; font-size: 13px; text-transform: uppercase;
            letter-spacing: .08em; }
.panel p { margin: 0; color: var(--muted); font-size: 14px; }
.panel.does h3 { color: #2b6cb0; }
.panel.doesnot { border-color: var(--warn-bg); border-width: 2px; }
.panel.doesnot h3 { color: var(--warn-bg); }
.panel.doesnot p { color: var(--ink); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
.scroll { overflow-x: auto; }
th, td { text-align: left; padding: 7px 10px;
         border-bottom: 1px solid var(--line); white-space: nowrap; }
th { font-size: 11px; text-transform: uppercase; letter-spacing: .07em;
     color: var(--muted); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
code, .mono { font-family: ui-monospace, "SF Mono", Menlo, monospace;
              font-size: 12.5px; }
.card { background: var(--card); border: 1px solid var(--line);
        border-radius: 11px; margin: 16px 0; overflow: hidden; }
.card > header { display: flex; align-items: center; gap: 12px;
                 padding: 13px 17px; border-bottom: 1px solid var(--line);
                 flex-wrap: wrap; }
.key { width: 32px; height: 32px; border-radius: 8px; flex: none;
       display: grid; place-items: center; font-weight: 800;
       background: var(--warn-bg); color: var(--warn-ink); }
.card h3 { margin: 0; font-size: 16px; flex: 1 1 240px; }
.pill { font-size: 11px; font-weight: 800; letter-spacing: .09em;
        padding: 4px 10px; border-radius: 999px; }
.badge { font-size: 10px; font-weight: 800; letter-spacing: .1em;
         padding: 4px 8px; border-radius: 4px;
         background: var(--warn-bg); color: var(--warn-ink); }
.body { padding: 15px 17px; }
dl { display: grid; grid-template-columns: 124px 1fr; gap: 5px 14px;
     margin: 0 0 14px; font-size: 14px; }
dt { color: var(--muted); font-size: 11px; text-transform: uppercase;
     letter-spacing: .07em; padding-top: 3px; padding-right: 8px; }
@media (max-width: 620px) { dl { grid-template-columns: 1fr; gap: 2px; }
  dt { padding-top: 8px; } }
dd { margin: 0; }
.reads { margin: 14px 0 0; padding: 11px 14px; border-left: 3px solid var(--warn-bg);
         background: rgba(128,128,128,.07); font-size: 14px; }
.contra { margin-top: 14px; border: 2px solid var(--warn-bg);
          border-radius: 8px; padding: 12px 14px; }
.contra h4 { margin: 0 0 8px; font-size: 12px; letter-spacing: .08em;
             text-transform: uppercase; color: var(--warn-bg); }
.refused { color: var(--warn-bg); font-weight: 700; }
footer { margin-top: 40px; padding-top: 18px; border-top: 2px solid var(--warn-bg);
         color: var(--muted); font-size: 13px; }
.stamp { display: inline-block; border: 2px solid var(--warn-bg);
         color: var(--warn-bg); border-radius: 6px; padding: 3px 9px;
         font-weight: 800; letter-spacing: .1em; font-size: 11px;
         transform: rotate(-1.5deg); }
"""


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _decision_table(decisions: list[dict]) -> str:
    if not decisions:
        return "<p class='mono'>no decision recorded</p>"
    rows = []
    for decision in decisions:
        action = (
            "merged" if decision["merged"]
            else "<span class='refused'>REFUSED</span>"
        )
        rows.append(
            "<tr>"
            f"<td class='mono'>{_esc(decision['address_a'])}</td>"
            f"<td class='mono'>{_esc(decision['address_b'])}</td>"
            f"<td class='mono'>{_esc(decision['verdict'])}</td>"
            f"<td class='num mono'>{decision['pooled_a']}/{decision['pooled_b']}</td>"
            f"<td class='num mono'>{decision['p_value']:.2e}</td>"
            f"<td class='num mono'>{decision['effect']:.4f}</td>"
            f"<td>{action}</td>"
            "</tr>"
        )
    return (
        "<div class='scroll'><table><thead><tr>"
        "<th>side A</th><th>side B</th><th>verdict</th>"
        "<th class='num'>pooled</th><th class='num'>p</th>"
        "<th class='num'>effect</th><th>action</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _scenario_card(scenario: dict, provenance: str) -> str:
    outcome = scenario["outcome"]
    ink, back = OUTCOME_STYLE.get(outcome, ("#2d3748", "#e2e8f0"))
    # The badge is driven by the payload's own flag, not by the template.
    badge = (
        f"<span class='badge'>{_esc(provenance.replace('_', ' '))}</span>"
        if scenario.get("demo") is True
        else ""
    )
    decisions = _decision_table(scenario["decisions"])

    contradiction = scenario.get("contradiction")
    contra_html = ""
    if contradiction:
        contra_html = (
            "<div class='contra'><h4>contradiction found after clustering</h4>"
            "<div class='scroll'><table><tbody>"
            f"<tr><td>one member against the other "
            f"{contradiction['component_size'] - 1} in its component</td>"
            f"<td class='mono'>{_esc(contradiction['verdict'])}</td>"
            f"<td class='num mono'>pooled {contradiction['pooled_member']}/"
            f"{contradiction['pooled_rest']}</td>"
            f"<td class='num mono'>p={contradiction['p_value']:.2e}</td>"
            f"<td class='num mono'>effect={contradiction['effect']:.4f}</td>"
            "</tr></tbody></table></div>"
            f"<p style='margin:9px 0 0'><strong>found by</strong> "
            f"{_esc(contradiction['found_by'])}<br>"
            f"<strong>action</strong> {_esc(contradiction['resolution'])}</p>"
            "</div>"
        )

    pooled = ", ".join(
        f"{_esc(entry['address'])} {entry['observations']}"
        for entry in scenario["pooled"]
    )
    return f"""
<article class="card">
  <header>
    <div class="key">{_esc(scenario['key'])}</div>
    <h3>{_esc(scenario['title'])}</h3>
    {badge}
    <span class="pill" style="background:{back};color:{ink}">{_esc(outcome)}</span>
  </header>
  <div class="body">
    <dl>
      <dt>situation</dt><dd>{_esc(scenario['situation'])}</dd>
      <dt>chain</dt><dd>{_esc(scenario['chain_story'])}</dd>
      <dt>network</dt><dd>{_esc(scenario['network_story'])}</dd>
      <dt>meaning</dt><dd>{_esc(OUTCOME_GLOSS.get(outcome, ''))}</dd>
      <dt>announcements</dt><dd class="mono">{scenario['announcements_seen']} seen,
        {scenario['usable_observations']} usable &nbsp;|&nbsp; pooled: {pooled}</dd>
      <dt>components</dt><dd class="mono">chain-only
        {scenario['chain_only_components']} &rarr; fused
        {scenario['fused_components']}</dd>
    </dl>
    {decisions}
    {contra_html}
    <p class="reads">{_esc(scenario['reads'])}</p>
  </div>
</article>"""


def render_html(envelope: dict) -> str:
    """The whole demonstration as one self-contained page.

    Raises :class:`~obsidianchain.demo.api.DemoFlagError` on an unflagged
    payload, before anything is rendered.
    """
    assert_demo_flagged(envelope)
    banner = envelope["banner"]
    provenance = envelope["provenance"]
    fixture, rule, totals = envelope["fixture"], envelope["rule"], envelope["totals"]

    summary_rows = "".join(
        "<tr>"
        f"<td class='mono'><strong>{_esc(s['key'])}</strong></td>"
        f"<td>{_esc(s['title'])}</td>"
        f"<td class='mono'>{_esc(s['outcome'])}</td>"
        f"<td class='mono'>{_esc(s['verdict'])}</td>"
        f"<td class='num mono'>{_headline_numbers(s)['pooled']}</td>"
        f"<td class='num mono'>{_headline_numbers(s)['p']}</td>"
        f"<td class='num mono'>{_headline_numbers(s)['effect']}</td>"
        "</tr>"
        for s in envelope["scenarios"]
    )
    cards = "".join(
        _scenario_card(s, provenance) for s in envelope["scenarios"]
    )
    fixture_rows = "".join(
        f"<tr><td>{_esc(label)}</td><td class='num mono'>{_esc(value)}</td></tr>"
        for label, value in (
            ("addresses", f"{fixture['addresses']:,}"),
            ("co-spend edges", f"{fixture['cospend_edges']:,}"),
            ("chain transactions", f"{fixture['chain_transactions']:,}"),
            ("announced", f"{fixture['announced_transactions']:,}"),
            ("never announced (scenario A)",
             f"{fixture['unannounced_transactions']:,}"),
            ("announcement records", f"{fixture['announcement_records']:,}"),
            ("usable after classification",
             f"{fixture['usable_transactions']:,}"),
            ("discarded NO_EVIDENCE", f"{fixture['discarded_no_evidence']:,}"),
            ("observers", f"{fixture['observers']:,}"),
            ("sigma", f"{fixture['sigma']:.4f} ({fixture['sigma_source']})"),
            ("seed", fixture["seed"]),
            ("observations sha256", fixture["dataset_sha256"][:24] + "..."),
        )
    )
    rule_rows = "".join(
        f"<tr><td>{_esc(label)}</td><td class='num mono'>{_esc(value)}</td></tr>"
        for label, value in (
            ("min pooled observations per side", rule["min_pooled_observations"]),
            ("min per-observer observations", rule["min_observer_observations"]),
            ("alpha", f"{rule['alpha']:.0e}"),
            ("min effect", f"{rule['min_effect']:.2f}"),
            ("unchanged from production",
             "yes" if rule["unchanged_from_production"] else "NO"),
        )
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>{_esc(banner)} - ObsidianChain constraint mechanism</title>
<style>{_CSS}</style></head>
<body>
<div class="hazard">{_esc(banner)}
  <small>Generated fixture, deterministic from seed {_esc(envelope['seed'])}.
  Every figure below is synthetic. None of it is a measurement of Bitcoin.</small>
</div>
<main class="wrap">
  <h1>The constraint mechanism, end to end</h1>
  <p class="sub">Five scenarios, one engine run, one chain.
    Provenance <code>{_esc(provenance)}</code> &middot; namespace
    <code>{_esc(envelope['namespace'])}</code></p>

  <div class="two">
    <section class="panel does"><h3>What this shows</h3>
      <p>{_esc(envelope['statements']['mechanism'])}</p></section>
    <section class="panel doesnot"><h3>What this does not show</h3>
      <p>{_esc(envelope['statements']['frozen_dataset'])}</p></section>
  </div>

  <h2>Summary</h2>
  <div class="panel scroll"><table><thead><tr>
    <th></th><th>scenario</th><th>outcome</th><th>verdict</th>
    <th class="num">pooled</th><th class="num">p</th><th class="num">effect</th>
  </tr></thead><tbody>{summary_rows}</tbody></table></div>

  <h2>Scenarios</h2>
  {cards}

  <h2>The rule, and the fixture it ran on</h2>
  <div class="two">
    <section class="panel"><h3>Rule (production, unchanged)</h3>
      <div class="scroll"><table><tbody>{rule_rows}</tbody></table></div>
      <p style="margin-top:10px">{_esc(rule['note'])}</p>
      <p style="margin-top:8px">{_esc(rule['cannot_link_only'])}</p>
    </section>
    <section class="panel"><h3>Fixture</h3>
      <div class="scroll"><table><tbody>{fixture_rows}</tbody></table></div>
    </section>
  </div>

  <h2>Totals</h2>
  <div class="panel scroll"><table><tbody>
    <tr><td>clusters, chain-only</td>
        <td class="num mono">{totals['chain_only_clusters']}</td></tr>
    <tr><td>clusters, fused (veto enabled)</td>
        <td class="num mono">{totals['fused_clusters']}</td></tr>
    <tr><td>merges blocked</td>
        <td class="num mono">{totals['merges_blocked']}</td></tr>
    <tr><td>components contested</td>
        <td class="num mono">{totals['components_contested']}</td></tr>
    <tr><td>decisions evaluated</td>
        <td class="num mono">{totals['decisions_evaluated']}</td></tr>
    <tr><td>decisions abstained</td>
        <td class="num mono">{totals['decisions_abstained']}</td></tr>
    <tr><td>every scenario behaved as described</td>
        <td class="num mono">
        {'yes' if totals['all_scenarios_as_described'] else 'NO'}</td></tr>
  </tbody></table></div>

  <footer>
    <p><span class="stamp">{_esc(banner)}</span></p>
    <p>Fixtures and outputs live under <code>{_esc(envelope['namespace'])}</code>
      and nowhere else. The frozen production dataset, its hash and every
      Phase&nbsp;1&ndash;3 number are untouched by this page; a test asserts
      it.</p>
    <p>All network data in this project is synthetic. It demonstrates the
      mechanism and validates nothing. Real validation needs mainnet capture
      with controlled ground truth.</p>
  </footer>
</main>
</body></html>
"""


def write_html(envelope: dict, path) -> Path:
    """Render and write the page. Refuses an unflagged payload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(envelope), encoding="utf-8")
    return path
