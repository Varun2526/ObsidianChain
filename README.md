# obsidianchain

Offline Bitcoin forensics prototype — NTRO problem statement **PS 26146**.

Address clustering over the [Elliptic++](https://github.com/git-disl/EllipticPlusPlus)
dataset, with network-layer evidence used to constrain the merge step.

**The core idea:** co-spend heuristics merge addresses transitively, so a
single wrong link welds two unrelated wallets together and the error
propagates until you get a super-cluster spanning half the graph.
Network-layer evidence — which peer announced a transaction, and when —
gives independent grounds to *refuse* a merge, blocking the false address
merges that produce those super-clusters.

## The offline constraint

This is an air-gapped tool. It is designed to run on an isolated forensic
workstation with no route to the internet, and the build is structured so
that constraint is enforced rather than assumed:

The precise guarantee is:

> Once the `python:3.11-slim` base image and the vendored artifacts are
> present locally, the application image can be built and run with Docker
> networking disabled.

That is deliberately narrower than "air-gapped from a fresh machine". Two
inputs still have to cross the network **once**, on a machine that has it:
the base image (`docker pull python:3.11-slim`) and the vendored artifacts
(`make vendor`). Everything after that is offline.

- Every container runs with `--network none`, which removes the network
  interface entirely — no DNS, no host gateway, no loopback to the outside.
- Dependencies install from an architecture-specific vendor directory —
  Python wheels plus the one system library the slim base image lacks
  (`libgomp1`, OpenMP, needed by lightgbm and scikit-learn) as a `.deb`.
  `docker build` runs with `--network none` and fails loudly if the vendor
  directory is missing, empty, or holds the wrong architecture, rather than
  falling back to PyPI or apt.
- No dependency makes network calls at runtime, and none needs a GPU.
- `PYTHONHASHSEED=0` and single-threaded BLAS are pinned in the image, and
  every dependency version — transitive included — is pinned in
  `requirements.txt`, so runs are reproducible.

`make isolation` demonstrates the runtime half: the CLI probes for a route
and exits non-zero if it finds one.

## Quick start

Requires Docker. The container is always Linux; the host OS does not need
to run Python.

```bash
make vendor   # ONCE per architecture, with internet
make build    # builds the image, docker build --network none
make verify   # checks data/raw for the dataset
make test     # pytest, air-gapped
make run      # runs the CLI, air-gapped
```

Only `make vendor` touches the network. After it, you can disconnect.

`make help` lists every target. `make shell` drops you into an interactive
container, also air-gapped. Pass CLI arguments with `ARGS`:

```bash
make run ARGS="info"
```

## Architecture

The primary target is **`linux/amd64`** — the team and SIH target — and it
is the default no matter what host you build on:

```
PLATFORM ?= linux/amd64
```

Three things have to agree, or the build is wrong: the Docker platform,
the Python wheels, and the system `.deb`. They all derive from `PLATFORM`
alone, so they cannot drift apart:

| `PLATFORM` | vendor directory | wheels | `.deb` |
|---|---|---|---|
| `linux/amd64` *(default)* | `vendor/linux-amd64/` | `*_x86_64.whl` | `*_amd64.deb` |
| `linux/arm64` | `vendor/linux-arm64/` | `*_aarch64.whl` | `*_arm64.deb` |

`make check-vendor` (a prerequisite of `make build`) asserts the vendored
artifacts match the selected platform, and the Dockerfile independently
checks BuildKit's `TARGETARCH` against the vendored `.deb` so a bare
`docker build` cannot pick the wrong set either.

`make vendor` never downloads on the host. It runs `pip download` and
`apt-get download` **inside a `PLATFORM` container**, so the wheels and
debs it fetches are by construction the right architecture — macOS wheels
are impossible because pip never runs on macOS.

### Apple silicon development

Building the default `linux/amd64` image on an M-series Mac works, via
Docker Desktop's emulation. It is correct but slow. For fast native
iteration, keep a second vendor set:

```bash
make vendor PLATFORM=linux/arm64
make build  PLATFORM=linux/arm64
make test   PLATFORM=linux/arm64
```

Both vendor sets coexist, so switching back costs nothing. Verify what you
actually built with `make arch` — host architecture and container
architecture are not the same thing:

```bash
make arch                        # amd64 image (emulated on Apple silicon)
make arch PLATFORM=linux/arm64   # arm64 image (native)
```

### Windows teammates

Windows does not need to run the application; Docker Desktop supplies the
Linux container. Use **WSL2**, which is Docker Desktop's usual backend and
provides the `make` that Windows lacks:

1. Install Docker Desktop with the WSL2 backend, and enable integration
   for your distribution in *Settings → Resources → WSL Integration*.
2. Clone **inside** the WSL2 filesystem (e.g. `~/dev/obsidianchain`), not
   under `/mnt/c`. This keeps bind-mount paths and file permissions sane.
3. `sudo apt install make` if needed, then:

```bash
make vendor
make build
make test
```

No `PLATFORM` override — amd64 is the default and is native there, so no
emulation is involved. `.gitattributes` pins LF line endings so a Windows
checkout cannot corrupt the `Makefile` or `Dockerfile`.

## Distributing vendor artifacts

`vendor/` is gitignored: it is ~140 MB per architecture and consists of
third-party binaries that do not belong in source control. Each teammate
therefore runs `make vendor` once, on a networked machine. It is
reproducible — `requirements.txt` pins every version, transitive
dependencies included — so everyone gets the same wheel set.

For a genuinely offline demo machine, copy the prepared directory across
by hand (USB, internal share):

```bash
# on a networked machine of the SAME architecture
make vendor PLATFORM=linux/amd64
tar czf obsidianchain-vendor-linux-amd64.tar.gz vendor/linux-amd64

# on the offline machine, alongside a pre-pulled python:3.11-slim
tar xzf obsidianchain-vendor-linux-amd64.tar.gz
make build
```

The base image must also be present there — `docker pull python:3.11-slim`
beforehand, or `docker save`/`docker load` it the same way.

## Getting the data

The dataset is **not** in this repo and is never downloaded by the code.
Fetch it yourself on a networked machine:

1. Get the Elliptic++ dataset from
   <https://github.com/git-disl/EllipticPlusPlus> — follow that repo's
   instructions for the dataset download links.
2. Copy the CSVs into `./data/raw/` on the host (a subdirectory is fine;
   the verifier searches recursively).
3. Run `make verify`.

`make verify` lists what it finds, reports row and column counts per CSV,
and prints a PASS/FAIL summary (exiting non-zero on FAIL, so it can gate a
pipeline). It specifically insists on
**`AddrAddr_edgelist.csv`** — without address-level edges there is no
co-spend clustering to do, and the transaction-only Elliptic dataset is
not a substitute.

`./data` is bind-mounted to `/data` in the container. Both `data/raw/` and
`data/processed/` are gitignored.

## Layout

```
src/obsidianchain/
├── io/          loaders, normalisation
├── cluster/     union-find, constrained clustering
├── network/     synthetic generator, later the observer
├── eval/        metrics
└── cli.py       entry point
scripts/verify_dataset.py   dataset presence check (stdlib only, offline)

vendor/                     gitignored, produced by `make vendor`
├── linux-amd64/            wheels + deb/ for the primary target
└── linux-arm64/            wheels + deb/ for Apple silicon dev
```

Status: scaffolding only. No clustering logic is implemented yet.
