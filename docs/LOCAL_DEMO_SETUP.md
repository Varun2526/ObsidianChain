# Local Demonstration & Clean Environment Setup

This document describes how to initialize a clean demonstration environment for **ObsidianChain** .

---

## 1. Resetting the Demonstration Database

To initialize a reproducible, clean state with zero prior development cases and clean schema:

```bash
make demo-reset
```

or via the CLI directly:

```bash
python3 -m obsidianchain.cli demo-reset --confirm
```

### What `demo-reset` Does
1. **Wipes Mutable Application State**: Clears `data/obsidianchain.sqlite3`, WAL/SHM files, and any previously uploaded datasets in `data/uploads/`.
2. **Applies All Schema Migrations**: Runs migrations v1 through v4 automatically.
3. **Provisions Clean Institutional Accounts**:
   - `admin` (Role: `ADMIN`, display: `System Administrator`)
   - `investigator` (Role: `INVESTIGATOR`, display: `Lead Investigator`)
   - `reviewer` (Role: `REVIEWER`, display: `Quality Reviewer`)
4. **Preserves Analytical Artifacts**: **Never** deletes or mutates the frozen analytical models or research parquet files in `data/raw/` or `data/processed/`.

---

## 2. Dynamic Temporary Credentials

To maintain strict security and avoid hardcoded secrets in the repository, `demo-reset` generates high-entropy temporary passwords at execution time and displays them in stdout:

```text
================================================================
OBSIDIANCHAIN — CLEAN DEMO DATABASE INITIALIZED
================================================================
Database:   Clean (schema v4)
Casework:   0 cases, 0 uploaded datasets
Analytics:  Preserved untouched (data/processed)
----------------------------------------------------------------
ROLE           USERNAME         TEMPORARY PASSWORD
----------------------------------------------------------------
ADMIN          admin            <dynamically_generated>
INVESTIGATOR   investigator     <dynamically_generated>
REVIEWER       reviewer         <dynamically_generated>
================================================================
```

### Setting Predefined Passwords for Local Walkthroughs

If desired for local automated evaluation, passwords can be supplied via environment variables before running the reset:

```bash
export OBSIDIANCHAIN_DEMO_ADMIN_PASSWORD="YourSecureAdminPassword123!"
export OBSIDIANCHAIN_DEMO_INVESTIGATOR_PASSWORD="YourSecureInvestigatorPassword123!"
export OBSIDIANCHAIN_DEMO_REVIEWER_PASSWORD="YourSecureReviewerPassword123!"

make demo-reset
```

---

## 3. Starting the Server

1. **Start Backend API Server**:
   ```bash
   python3 -m uvicorn obsidianchain.api.app:app --host 127.0.0.1 --port 8000
   ```

2. **Start Frontend Development Server**:
   ```bash
   npm --prefix frontend run dev
   ```

3. Open `http://localhost:5173/` in your browser and log in with the generated credentials.
