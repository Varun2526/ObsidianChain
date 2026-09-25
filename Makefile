# obsidianchain - build and run, air-gapped.
#
# One-time, online:   make vendor
# Everything else runs with the container's network interface removed.

.DEFAULT_GOAL := help
SHELL := /bin/sh

IMAGE       ?= obsidianchain
TAG         ?= latest
DATA_DIR    ?= $(CURDIR)/data
VENDOR_ROOT ?= $(CURDIR)/vendor
PY_BASE     ?= python:3.11-slim
ARGS        ?=

# Native libraries the wheels link against but the slim base image omits.
# libgomp1 is OpenMP, required by lightgbm and by parts of scikit-learn.
SYS_DEBS    ?= libgomp1

# The team/SIH target is Linux AMD64, so that is the default regardless of
# the host. Apple silicon developers who want a fast native image can build
# with PLATFORM=linux/arm64; the two vendor sets live side by side.
#
#   make vendor PLATFORM=linux/arm64
#   make build  PLATFORM=linux/arm64
#
PLATFORM    ?= linux/amd64

# Everything architecture-specific derives from PLATFORM alone, so the
# Docker platform, the Python wheels and the .deb can never disagree.
#   linux/amd64 -> vendor/linux-amd64, dpkg "amd64", wheels tagged x86_64
#   linux/arm64 -> vendor/linux-arm64, dpkg "arm64",  wheels tagged aarch64
VENDOR_ARCH := $(subst /,-,$(PLATFORM))
DEB_ARCH    := $(lastword $(subst /, ,$(PLATFORM)))
VENDOR_REL  := vendor/$(VENDOR_ARCH)
VENDOR_DIR  := $(VENDOR_ROOT)/$(VENDOR_ARCH)

ifeq ($(DEB_ARCH),amd64)
  WHEEL_ARCH := x86_64
else ifeq ($(DEB_ARCH),arm64)
  WHEEL_ARCH := aarch64
else
  $(error Unsupported PLATFORM "$(PLATFORM)" - use linux/amd64 or linux/arm64)
endif

# --network none removes the container's network interface entirely.
# There is no route out, no DNS, no host gateway. This is the demo: the
# tooling cannot phone home even if it wanted to.
OFFLINE := --network none

DOCKER_RUN := docker run --rm $(OFFLINE) --platform $(PLATFORM) \
	-v $(DATA_DIR):/data \
	-e OBSIDIANCHAIN_DATA=/data

.PHONY: help vendor build run shell test verify isolation arch freeze dirs clean demo serve \
        clean-vendor clean-vendor-all check-vendor demo-reset web deploy undeploy

help: ## Show this help
	@echo "obsidianchain - offline Bitcoin forensics prototype"
	@echo ""
	@echo "  platform: $(PLATFORM)   image: $(IMAGE):$(TAG)"
	@echo "  vendor:   $(VENDOR_REL)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Only 'vendor' touches the network. Everything else runs --network none."

# pip and apt run INSIDE a $(PLATFORM) container, so they resolve wheels and
# debs for that architecture. Nothing is ever downloaded by the host, which is
# what makes macOS wheels structurally impossible here.
vendor: ## [ONLINE, run once per arch] Download wheels + debs for PLATFORM
	@echo ">> This is the only target that uses the network."
	@echo ">> Target: $(PLATFORM) -> $(VENDOR_REL)"
	@mkdir -p $(VENDOR_DIR)/deb
	docker run --rm --platform $(PLATFORM) \
		-e PIP_NO_CACHE_DIR=1 \
		-v $(CURDIR):/w -w /w \
		$(PY_BASE) \
		sh -c 'set -eu; \
		  echo "   container arch: $$(uname -m) / $$(dpkg --print-architecture)"; \
		  apt-get update -qq; \
		  (cd $(VENDOR_REL)/deb && apt-get download $(SYS_DEBS)); \
		  pip download -r requirements.txt -d $(VENDOR_REL) --only-binary=:all:; \
		  chown -R $$(stat -c %u:%g /w) vendor'
	@echo ">> Vendored $$(ls -1 $(VENDOR_DIR)/*.whl | wc -l | tr -d ' ') wheels" \
		"and $$(ls -1 $(VENDOR_DIR)/deb/*.deb | wc -l | tr -d ' ') system packages" \
		"into $(VENDOR_REL)."
	@echo ">> You can disconnect from the network now."

check-vendor: ## Assert vendored artifacts exist and match PLATFORM
	@if [ ! -d "$(VENDOR_DIR)" ] || [ -z "$$(ls -A $(VENDOR_DIR) 2>/dev/null)" ]; then \
		echo "ERROR: $(VENDOR_REL) is empty or missing."; \
		echo "       Run 'make vendor PLATFORM=$(PLATFORM)' on a networked machine."; \
		exit 1; \
	fi
	@if [ -z "$$(ls -1 $(VENDOR_DIR)/deb/*.deb 2>/dev/null)" ]; then \
		echo "ERROR: $(VENDOR_REL)/deb has no .deb files."; \
		echo "       Run 'make vendor PLATFORM=$(PLATFORM)'."; \
		exit 1; \
	fi
	@# Cross-architecture artifacts would otherwise fail deep inside the
	@# build with a confusing dpkg/pip error. Catch it here instead.
	@bad_deb=$$(ls -1 $(VENDOR_DIR)/deb/*.deb | grep -v "_$(DEB_ARCH)\.deb$$" || true); \
	if [ -n "$$bad_deb" ]; then \
		echo "ERROR: $(VENDOR_REL) holds .deb files that are not $(DEB_ARCH):"; \
		echo "$$bad_deb" | sed 's|^|         |'; \
		echo "       Re-run 'make vendor PLATFORM=$(PLATFORM)'."; \
		exit 1; \
	fi
	@if ! ls -1 $(VENDOR_DIR)/*.whl | grep -q "$(WHEEL_ARCH)"; then \
		echo "ERROR: no $(WHEEL_ARCH) wheels found in $(VENDOR_REL)."; \
		echo "       These wheels do not match PLATFORM=$(PLATFORM)."; \
		echo "       Re-run 'make vendor PLATFORM=$(PLATFORM)'."; \
		exit 1; \
	fi
	@echo ">> vendor OK: $(VENDOR_REL) matches $(PLATFORM) ($(WHEEL_ARCH)/$(DEB_ARCH))"

web: ## Build the investigation console (frontend/dist) on the host
	npm --prefix frontend ci --no-audit --no-fund
	npm --prefix frontend run build

build: check-vendor ## Build the image offline for PLATFORM (run `make web` first)
	@test -f frontend/dist/index.html || { echo "ERROR: frontend/dist missing - run 'make web' first."; exit 1; }
	docker build \
		--platform $(PLATFORM) \
		--network none \
		--build-arg VENDOR_PATH=$(VENDOR_REL) \
		-t $(IMAGE):$(TAG) .
	@echo ">> Built $(IMAGE):$(TAG) for $(PLATFORM)"

run: dirs ## Run the CLI, air-gapped (ARGS="info")
	$(DOCKER_RUN) $(IMAGE):$(TAG) $(ARGS)

shell: dirs ## Interactive shell in the container, air-gapped
	docker run --rm -it $(OFFLINE) --platform $(PLATFORM) \
		-v $(DATA_DIR):/data -e OBSIDIANCHAIN_DATA=/data \
		--entrypoint /bin/bash $(IMAGE):$(TAG)

test: dirs ## Run pytest inside the container, air-gapped
	$(DOCKER_RUN) --entrypoint pytest $(IMAGE):$(TAG) /app/tests

# The demonstration is SYNTHETIC and confined to data/demo. It never reads or
# writes data/processed, and a test asserts that.
# The API is read-only over precomputed artifacts. --network none is
# dropped here and ONLY here among the run targets, because a server that
# cannot accept a connection is not a server; the port is published to
# localhost only.
serve: dirs ## Serve the read-only API on localhost:8000
	docker run --rm -it --platform $(PLATFORM) \
		-p 127.0.0.1:8000:8000 \
		-v $(DATA_DIR):/data -e OBSIDIANCHAIN_DATA=/data \
		$(IMAGE):$(TAG) serve --host 0.0.0.0 --port 8000

# A long-running deployment: detached, restarted on failure, health-checked,
# console and API on one origin. BIND defaults to all interfaces so the
# instance is reachable from other machines on the network; set
# BIND=127.0.0.1 to keep it local.
BIND ?= 0.0.0.0
PORT ?= 8080
deploy: dirs ## Run the console + API detached on $(BIND):$(PORT)
	-docker rm -f obsidianchain >/dev/null 2>&1
	docker run -d --name obsidianchain --restart unless-stopped --platform $(PLATFORM) \
		-p $(BIND):$(PORT):8000 \
		-v $(DATA_DIR):/data -e OBSIDIANCHAIN_DATA=/data \
		$(IMAGE):$(TAG) serve --host 0.0.0.0 --port 8000
	@echo ">> deployed: http://<this-host>:$(PORT)/  (docker ps; docker logs obsidianchain)"

undeploy: ## Stop and remove the deployment container
	docker rm -f obsidianchain

demo: dirs ## Run the five DEMO scenarios; writes data/demo/output/index.html
	$(DOCKER_RUN) $(IMAGE):$(TAG) demo --rebuild
	@echo ""
	@echo ">> open $(DATA_DIR)/demo/output/index.html"

demo-reset: dirs ## Deterministically reset the casework database and generate clean demo credentials
	python3 -m obsidianchain.cli demo-reset --confirm --data-root $(DATA_DIR)


verify: dirs ## Check data/raw for the Elliptic++ dataset
	$(DOCKER_RUN) --entrypoint python $(IMAGE):$(TAG) /app/scripts/verify_dataset.py

isolation: ## Prove the container has no network interface
	$(DOCKER_RUN) $(IMAGE):$(TAG) isolation

arch: ## Report the architecture of the built image and its container
	@echo "requested        : $(PLATFORM)"
	@echo "image arch       : $$(docker image inspect $(IMAGE):$(TAG) \
		--format '{{.Os}}/{{.Architecture}}')"
	@echo "container uname  : $$(docker run --rm $(OFFLINE) \
		--platform $(PLATFORM) --entrypoint uname $(IMAGE):$(TAG) -m)"
	@echo "container python : $$(docker run --rm $(OFFLINE) \
		--platform $(PLATFORM) --entrypoint python $(IMAGE):$(TAG) -V)"

freeze: ## Print pinned versions derived from the vendored wheels
	@ls -1 $(VENDOR_DIR)/*.whl 2>/dev/null \
		| sed 's|.*/||' \
		| awk -F- '{gsub(/_/,"-",$$1); print $$1 "==" $$2}' \
		| sort -u \
		|| echo "$(VENDOR_REL) is empty - run 'make vendor'"

dirs:
	@mkdir -p $(DATA_DIR)/raw $(DATA_DIR)/processed $(DATA_DIR)/uploads

# Deliberately leaves $(DATA_DIR)/obsidianchain.sqlite3 and
# $(DATA_DIR)/uploads alone: those hold investigator state and the
# datasets uploaded into cases, which must survive regenerating the
# analytical artifacts. A case outlives the run it references.
clean: ## Remove caches and processed outputs
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache *.egg-info src/*.egg-info
	find $(DATA_DIR)/processed -type f ! -name .gitkeep -delete 2>/dev/null || true

clean-vendor: ## Delete vendored artifacts for PLATFORM (internet needed to restore)
	rm -rf $(VENDOR_DIR)

clean-vendor-all: ## Delete vendored artifacts for every architecture
	rm -rf $(VENDOR_ROOT)
