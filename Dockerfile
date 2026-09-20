# obsidianchain - offline Bitcoin forensics prototype (NTRO PS 26146)
#
# Once the python:3.11-slim base image and the vendored artifacts are
# present locally, this image builds with Docker networking disabled. All
# Python dependencies are installed from an architecture-specific vendor
# directory populated by `make vendor`; if it is missing or holds the wrong
# architecture the build fails loudly rather than reaching for PyPI or apt.
#
#   make vendor   # once per architecture, online
#   make build    # offline from here on
#   make run      # docker run --network none
#
# Primary target is linux/amd64 (the team/SIH target). linux/arm64 is also
# supported for Apple silicon development. The Docker platform, the Python
# wheels and the .deb must all agree; VENDOR_PATH is what keeps them
# aligned, and TARGETARCH below is checked against the vendored .deb.

FROM python:3.11-slim

# Which vendor directory to install from. Set by `make build` from
# PLATFORM; the default matches the primary target.
ARG VENDOR_PATH=vendor/linux-amd64

# Supplied automatically by BuildKit from --platform ("amd64" / "arm64").
ARG TARGETARCH

# Deterministic, quiet, no stray writes.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INDEX=1 \
    OBSIDIANCHAIN_DATA=/data \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# ---- dependencies, from vendored wheels only -------------------------
# --no-index forbids PyPI; --find-links points pip at the local wheels.
COPY ${VENDOR_PATH}/ /wheels/
COPY requirements.txt /app/requirements.txt
#
# libgomp1 (OpenMP) is required by lightgbm and by parts of scikit-learn,
# and is not present in the slim base image. `make vendor` downloads it as
# a .deb alongside the wheels so this stays an offline install rather than
# an apt-get over the network.
RUN set -eux; \
    if [ -z "$(ls -A /wheels 2>/dev/null)" ]; then \
        echo "ERROR: /wheels is empty - run 'make vendor' on a networked host first." >&2; \
        exit 1; \
    fi; \
    if [ -z "$(ls -A /wheels/deb/*.deb 2>/dev/null)" ]; then \
        echo "ERROR: /wheels/deb has no .deb files - re-run 'make vendor'." >&2; \
        exit 1; \
    fi; \
    if [ -n "${TARGETARCH:-}" ] && ! ls /wheels/deb/*_${TARGETARCH}.deb >/dev/null 2>&1; then \
        echo "ERROR: VENDOR_PATH=${VENDOR_PATH} does not hold ${TARGETARCH} packages." >&2; \
        echo "       Vendored artifacts do not match the build platform." >&2; \
        echo "       Run: make vendor PLATFORM=linux/${TARGETARCH}" >&2; \
        exit 1; \
    fi; \
    dpkg -i /wheels/deb/*.deb; \
    pip install --no-index --find-links=/wheels -r /app/requirements.txt; \
    rm -rf /wheels

# ---- the package itself ----------------------------------------------
# --no-build-isolation reuses the image's setuptools instead of fetching
# a fresh build environment from PyPI.
COPY pyproject.toml /app/pyproject.toml
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY research/ /app/research/
COPY tests/ /app/tests/
RUN pip install --no-index --no-build-isolation --no-deps /app

# ---- API port ---------------------------------------------------------
# Informational only; nothing listens unless `obsidianchain serve` is the
# command. The CLI binds 127.0.0.1 by default, so reaching it from outside
# the container needs an explicit --host as well as a published port.
EXPOSE 8000

# ---- data mount point -------------------------------------------------
# ./data is bind-mounted here at run time; see the Makefile.
RUN mkdir -p /data/raw /data/processed
VOLUME ["/data"]

# Fail fast if the image is ever run with a network attached by mistake.
# (Informational only - `docker run --network none` is what enforces it.)
LABEL org.opencontainers.image.title="obsidianchain" \
      org.opencontainers.image.description="Offline Bitcoin forensics prototype (NTRO PS 26146)" \
      org.obsidianchain.airgapped="true"

ENTRYPOINT ["obsidianchain"]
CMD ["info"]
