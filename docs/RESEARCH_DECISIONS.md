# Research → Production Decisions
**AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic**  
**Repository Layer:** `docs/RESEARCH_DECISIONS.md` (The Empirical Journey from Experiment to Production Safeguard)

---

## 1. Research Philosophy

ObsidianChain was built through rigorous empirical research rather than unverified assumptions. In high-stakes forensic analysis, speculative features and ungrounded heuristics lead to catastrophic false accusations and investigative dead ends.

Every production component in this platform traces back to a structured empirical loop:

$$\text{Research Question} \longrightarrow \text{Controlled Experiment} \longrightarrow \text{Empirical Observation} \longrightarrow \text{Engineering Decision} \longrightarrow \text{Production Safeguard}$$

This document records the empirical evidence, critical evaluations, and design decisions that shaped the final frozen platform.

---

## 2. Blockchain Clustering (Common-Input Heuristic)

### The Research Question
Can we reliably group distinct Bitcoin addresses into single entity clusters to evaluate entity-level behavior?

### Empirical Findings
1. **The Multi-Input Co-Spend Heuristic:** When a transaction spends UTXOs from multiple distinct input addresses, the spending party must possess the private keys for all participating inputs (in standard single-key transactions). By constructing a disjoint-set forest (Union-Find) over transaction input addresses, we can compute the transitive closure of shared control.
2. **Computational Complexity:** A naive clique approach ($k(k-1)/2$ edges per transaction) produces millions of redundant edges for large batch transactions. Connecting inputs as a star graph centered on the first input ($k-1$ edges) yields identical connected components while reducing edge count by over 90% and maintaining near-linear $O(\alpha(N))$ complexity via union-by-rank and path-compression.
3. **What It Establishes:** Co-spend establishes a strong cryptographic presumption of common control or coordinated signing.
4. **What It Does NOT Establish:**
   - It does not identify the real-world legal entity or natural person behind the keys.
   - It does not hold unconditionally for CoinJoin / Wasabi collaborative transactions, where multiple independent parties intentionally combine inputs into a single transaction to obscure ownership.
5. **Why Addr $\to$ Addr Money-Flow Edges Are NOT Merged:**  
   In naive graph representations, edges often represent value transfer from input address $A$ to output address $B$. **Treating value transfer as an entity merge would destroy the graph.** Value transfer represents a payment between different economic actors. Merging on money flow would instantly collapse the entire Bitcoin blockchain into a single monolithic super-cluster.

### Production Consequence
- Multi-input co-spend is implemented via star-topology Union-Find (`src/obsidianchain/cluster/unionfind.py`).
- Money-flow edges are treated as directed transactional flow in graph projections, strictly separated from entity resolution clustering.

---

## 3. Change-Address Heuristics

### The Research Question
Can we identify change outputs to link payment transactions back to the sender's entity?

### Empirical Findings & Cluster Collapse Risk
In standard Bitcoin transactions, a sender pays a recipient and returns the remaining change to a freshly generated address controlled by the sender's wallet.
- **Tested Heuristics:**
  1. *Self-reference:* An output address that also appears in the inputs. (Certain, but yields no new entity merges).
  2. *One-fresh output:* Exactly one output address is completely new to the blockchain, while the other has an established history.
  3. *Fresher first block:* The change output appears later in block history than the payment destination.
- **The Risk of Cluster Collapse:** Unlike multi-input co-spend, change identification is an *inferential guess*. If a change heuristic makes a single false positive merge on a high-volume merchant, exchange, or mining pool, Union-Find transitively unions all downstream customers into a catastrophic super-cluster.

### Production Decision
- Change detection was retained **strictly as a secondary, gated heuristic** (`src/obsidianchain/cluster/change.py`).
- It is gated to **strictly 2-output transactions** (excluding batch payments).
- Merge confidence must exceed an empirical threshold (default 0.70).
- Co-spend merges are applied *first*; change merges are applied *second* and accounted for in separate audit metrics (`change_merges`).

---

## 4. Temporal Analysis & Cluster Evolution

### The Research Question
How do address clusters evolve over time, and does static clustering introduce temporal lookahead bias?

### Empirical Findings
- Evaluating address behavior across time showed that clusters grow continuously as entities generate fresh addresses and consolidate UTXOs.
- If an entity's cluster size at block 600,000 is used to score a transaction that occurred at block 200,000, the model benefits from **future knowledge** that the investigator would not have possessed at that time.

### Production Decision
- All graph and cluster metrics are extracted **strictly as-of event timestamp $t$** (`cluster_size_asof_t`, `in_degree_asof_t`, `n_txs_asof_t`).
- The 17-stage pipeline asserts strict chronological integrity, preventing future graph growth from leaking into historical evaluation states.

---

## 5. Entity Resolution Validation

### The Research Question
How can we empirically validate entity clustering when complete ground-truth ownership is unavailable?

### Empirical Findings
- Using the Elliptic++ dataset, we evaluated candidate clusters against labeled wallet classes.
- We constructed positive and negative address pairs within connected components.
- Addresses linked by multi-input co-spend exhibited an overwhelming agreement in class labels (licit vs. illicit) compared to null-model random graph walks, confirming that co-spend partitions economic entities with high fidelity.
- However, boundary-spanning clusters occasionally exhibited mixed labels, particularly when interacting with darknet deposit addresses or peeling chains.

### Production Decision
- Clusters are reported as **candidate entities**, not absolute facts.
- The UI exposes the cluster size, internal transaction counts, and constituent addresses so investigators can visually verify cluster boundaries.

---

## 6. Network Telemetry & Controlled World Research

### The Research Question
Can P2P network packet observations reveal the physical identity or home location of a transaction's originator?

### Empirical Findings
Controlled network experiments and P2P gossip analysis established:
1. **Pervasive Gossip Relay:** Over 84% of transactions are announced to listening observers by multiple independent peers. An observer captures the peer that forwarded the gossip message, which is almost never the wallet client.
2. **Vantage Bias:** Two observers situated in different autonomous systems receive conflicting "first-seen" peer IPs for the identical transaction hash due to network latency and topology differences.
3. **Shared Infrastructure:** A single public Electrum server or VPN exit node broadcasts transactions for thousands of unrelated individuals across the globe.
4. **False Split & Merge Risk:** Merging wallets because they share an announcing IP creates catastrophic false merges. Splitting wallets because they announce from different IPs creates false fragmentation.

### Production Decision
- Network telemetry is classified strictly as `NETWORK_CONTEXT`.
- The system **never** equates IP address with wallet ownership.
- The platform issues explicit evidentiary warnings alongside all network observations.

---

## 7. Network Evidence: What Justifies a CANNOT-LINK?

### The Research Question
Under what circumstances does network evidence justify asserting that two transactions or clusters are *unrelated*?

### Empirical Findings
- An IP match does not justify a "MUST-LINK" (due to VPNs, NAT, and shared nodes).
- Conversely, does a complete divergence in network telemetry justify a "CANNOT-LINK"?
- Research demonstrated that even a single user with a laptop moves between home Wi-Fi, coffee shop networks, and mobile 5G, naturally broadcasting from different ASNs and countries over time.
- However, persistent, simultaneous transaction broadcasts across completely disjoint routing topologies over extended periods provide **investigative hypothesis separation**.

### Production Decision
- Network evidence is used to suggest candidate separation context, never automated mathematical impossibility.
- When network telemetry is inconclusive, the system exercises **honest abstention (`INSUFFICIENT_EVIDENCE`)** rather than forcing an unsubstantiated link.

---

## 8. Model Selection: Logistic Regression vs. LightGBM vs. Random Forest

### The Research Question
Which machine learning architecture provides the most reliable risk prioritization on heavily imbalanced Bitcoin transaction graphs?

### The Candidate Evaluation (Validation Split: Timesteps 35–41, Prevalence: 6.26%)
- **Logistic Regression (L2 penalty):**
  - Validation PR-AUC: **0.2654** | Precision@100: **55.0%** | Fit Time: **0.46s**
  - *Analysis:* Failed to model non-linear interactions between transaction amounts, fee ratios, and co-spend degree.
- **LightGBM (Gradient Boosted Trees):**
  - Validation PR-AUC: **0.5228** | Precision@100: **71.0%** | Fit Time: **1.49s**
  - *Analysis:* Highly performant, but produced lower precision in the critical top-100 tier due to aggressive gradient splits on minority clusters.
- **Random Forest (120 trees, max_depth=14, balanced subsample):**
  - Validation PR-AUC: **0.5702** | Precision@100: **99.0%** | Fit Time: **1.25s**
  - *Analysis:* **Winner.** Delivered the highest overall PR-AUC and an extraordinary **99.0% Precision@100**. Bagging across feature subsets provided superior resilience against heavy-tailed outliers and distribution shifts.

### Probability Calibration
- Uncalibrated Random Forest voting percentages tended to cluster between 0.2 and 0.8.
- Fitting an **Isotonic Regressor** exclusively on the validation split reduced Brier score from **0.04186 to 0.03612**, yielding true monotonic posterior probabilities.

### Production Decision
- Random Forest (120 trees, `max_depth=14`) was selected and **FROZEN** as the production analytical model.
- LightGBM was retired to historical research benchmarks (`research/`).

---

## 9. Research Findings That Became Production Safeguards

| Empirical Research Finding | Resulting Production Safeguard |
| :--- | :--- |
| Network IP matches often represent shared relays, not common owners. | Rule: Network IP overlap never triggers an entity cluster merge. |
| Future graph growth leaks into historical snapshots. | Feature engine calculates all features strictly as-of event timestamp $t$. |
| Extreme transaction amounts distort Gaussian distributions. | Adopted Median Absolute Deviation (MAD) robust Z-scoring. |
| Tree voting proportions diverge from empirical probabilities. | Applied Isotonic Regression calibration fit strictly on validation data. |
| Missing network telemetry is normal in full-node dumps. | Missing fields evaluate to `NaN`; never imputed as zero. |
| Research labels create circular analytical evaluation. | Implemented API boundary assertion blocking all ground-truth label keys. |
| Chain-of-custody in digital forensics requires auditability. | Engineered append-only SQLite audit ledger and Merkle bundle verification. |

---

## 10. Research That Was NOT Promoted to Production

To maintain system reliability, security, and explainability, several explored avenues were **deliberately rejected** for production:

1. **LightGBM as the Production Model:**  
   While fast, LightGBM achieved 71.0% Precision@100 compared to Random Forest's 99.0%, and demonstrated higher volatility across temporal boundaries. It was preserved in `research/` but excluded from production.
2. **Graph Neural Networks (GNNs / GCNs) & Transformers:**  
   GNNs (e.g., RGCN, EvolveGCN) require GPU acceleration, introduce large non-deterministic PyTorch dependencies, and produce opaque black-box node embeddings that cannot be explained to a court. They were rejected in favor of tabular features + Random Forest.
3. **Hard Network Identity Heuristics:**  
   Heuristics attempting to declare "Wallet X belongs to IP Y" were completely rejected due to gossip propagation physics, NAT, and VPN prevalence.
4. **Live Mainnet Network Sniffing:**  
   A live P2P network crawler was rejected as out-of-scope for an offline forensic analysis workbench. The system focuses on post-capture bulk analysis rather than volatile live wiretapping.
5. **Distributed Infrastructure (Kafka, PostgreSQL, Spark):**  
   Heavyweight distributed middleware was rejected. An offline investigation tool deployed in containerized field environments must run self-contained on a standard workstation with minimal memory footprint.
