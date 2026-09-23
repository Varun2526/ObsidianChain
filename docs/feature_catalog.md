# Feature catalog

Generated from `src/obsidianchain/contracts/features.py` (obsidianchain.feature_contract/1); engine schema `ps_native_features/5`. Do not edit by hand: run `python scripts/render_feature_catalog.py`.

Information sets: `TX_ITSELF` = this transaction only; `PRIOR_EVENTS` = events strictly before this one in (timestamp, event_order); `PRIOR_EVENTS_AND_TX` = both. No feature reads a label.

| feature | group | in model | information set | same-step behaviour | NaN means | range | notes |
|---|---|---|---|---|---|---|---|
| `input_count` | A_transaction | yes | TX_ITSELF | not applicable: a property of this transaction only | never (violation) | [0.0, None] |  |
| `output_count` | A_transaction | yes | TX_ITSELF | not applicable: a property of this transaction only | never (violation) | [0.0, None] |  |
| `total_input_amount` | A_transaction | yes | TX_ITSELF | not applicable: a property of this transaction only | never (violation) | [0.0, None] |  |
| `fee` | A_transaction | yes | TX_ITSELF | not applicable: a property of this transaction only | fee not supplied by the capture | [0.0, None] |  |
| `fee_ratio` | A_transaction | yes | TX_ITSELF | not applicable: a property of this transaction only | fee missing or zero input value (not measurable) | [0.0, 1.0] |  |
| `input_amount_mean` | A_transaction | yes | TX_ITSELF | not applicable: a property of this transaction only | never (violation) | [0.0, None] |  |
| `output_amount_mean` | A_transaction | yes | TX_ITSELF | not applicable: a property of this transaction only | never (violation) | [0.0, None] |  |
| `input_spread` | A_transaction | yes | TX_ITSELF | not applicable: a property of this transaction only | non-positive mean (not measurable) | [0.0, None] |  |
| `output_spread` | A_transaction | yes | TX_ITSELF | not applicable: a property of this transaction only | non-positive mean (not measurable) | [0.0, None] |  |
| `n_txs_asof_t` | B_address_history | yes | PRIOR_EVENTS | only events earlier in (timestamp, event_order) are counted; an event at the same point is not visible | never (violation) | [0.0, None] |  |
| `n_sent_asof_t` | B_address_history | yes | PRIOR_EVENTS | only events earlier in (timestamp, event_order) are counted; an event at the same point is not visible | never (violation) | [0.0, None] |  |
| `n_recv_asof_t` | B_address_history | yes | PRIOR_EVENTS | only events earlier in (timestamp, event_order) are counted; an event at the same point is not visible | never (violation) | [0.0, None] |  |
| `btc_recv_total_asof_t` | B_address_history | yes | PRIOR_EVENTS | only events earlier in (timestamp, event_order) are counted; an event at the same point is not visible | never (violation) | [0.0, None] | per-address amounts are even splits on Elliptic++ (source has no per-output values) |
| `net_flow_asof_t` | B_address_history | yes | PRIOR_EVENTS | only events earlier in (timestamp, event_order) are counted; an event at the same point is not visible | never (violation) | [None, None] | per-address amounts are even splits on Elliptic++ |
| `active_duration_seconds` | B_address_history | yes | PRIOR_EVENTS_AND_TX | only events earlier in (timestamp, event_order) are counted; an event at the same point is not visible | never (violation) | [0.0, None] | on Elliptic++ a multiple of one timestep (surrogate timestamps) |
| `gap_since_last_tx` | B_address_history | yes | PRIOR_EVENTS_AND_TX | only events earlier in (timestamp, event_order) are counted; an event at the same point is not visible | never (violation) | [0.0, None] | on Elliptic++ a multiple of one timestep (surrogate timestamps) |
| `unique_counterparties_asof_t` | C_graph | yes | PRIOR_EVENTS | only events earlier in (timestamp, event_order) are counted; an event at the same point is not visible | never (violation) | [0.0, None] |  |
| `cluster_size_asof_t` | C_graph | yes | PRIOR_EVENTS_AND_TX | only events earlier in (timestamp, event_order) are counted; an event at the same point is not visible | never (violation) | [1.0, None] |  |
| `is_peeling_candidate` | D_patterns | yes | TX_ITSELF | not applicable: a property of this transaction only | never (violation) | [0.0, 1.0] |  |
| `is_mixing_candidate` | D_patterns | yes | TX_ITSELF | not applicable: a property of this transaction only | never (violation) | [0.0, 1.0] |  |
| `addr_is_sender` | F_role | yes | TX_ITSELF | not applicable: a property of this transaction only | never (violation) | [0.0, 1.0] |  |
| `addr_is_self_change` | F_role | yes | TX_ITSELF | not applicable: a property of this transaction only | never (violation) | [0.0, 1.0] |  |
| `counterparty_max_n_txs_asof_t` | F_role | yes | PRIOR_EVENTS_AND_TX | only events earlier in (timestamp, event_order) are counted; an event at the same point is not visible | never (violation) | [0.0, None] |  |
| `counterparty_mean_n_txs_asof_t` | F_role | yes | PRIOR_EVENTS_AND_TX | only events earlier in (timestamp, event_order) are counted; an event at the same point is not visible | never (violation) | [0.0, None] |  |
| `upstream_funded_share` | G_upstream | yes | PRIOR_EVENTS_AND_TX | only events earlier in (timestamp, event_order) are counted; an event at the same point is not visible | never (violation) | [0.0, 1.0] |  |
| `upstream_mean_output_count` | G_upstream | yes | PRIOR_EVENTS | only events earlier in (timestamp, event_order) are counted; an event at the same point is not visible | no input has a strictly earlier funding transaction | [0.0, None] |  |
| `upstream_mean_input_count` | G_upstream | yes | PRIOR_EVENTS | only events earlier in (timestamp, event_order) are counted; an event at the same point is not visible | no input has a strictly earlier funding transaction | [0.0, None] |  |
| `upstream_peel_share` | G_upstream | yes | PRIOR_EVENTS | only events earlier in (timestamp, event_order) are counted; an event at the same point is not visible | no input has a strictly earlier funding transaction | [0.0, 1.0] |  |
| `upstream_mix_share` | G_upstream | yes | PRIOR_EVENTS | only events earlier in (timestamp, event_order) are counted; an event at the same point is not visible | no input has a strictly earlier funding transaction | [0.0, 1.0] |  |
| `upstream_min_hold_seconds` | G_upstream | yes | PRIOR_EVENTS_AND_TX | only events earlier in (timestamp, event_order) are counted; an event at the same point is not visible | no input has a strictly earlier funding transaction | [0.0, None] | 0 for same-step funding on Elliptic++ (surrogate timestamps) |
| `upstream_chain_depth` | G_upstream | yes | PRIOR_EVENTS_AND_TX | only events earlier in (timestamp, event_order) are counted; an event at the same point is not visible | never (violation) | [0.0, 10.0] |  |
| `network_observation_count` | E_network | no | TX_ITSELF | not applicable: a property of this transaction only | no observation of this txid | [0.0, None] |  |
| `observer_diversity` | E_network | no | TX_ITSELF | not applicable: a property of this transaction only | no observation of this txid | [0.0, None] |  |
| `peer_count` | E_network | no | TX_ITSELF | not applicable: a property of this transaction only | no observation of this txid | [0.0, None] |  |
| `asn_count` | E_network | no | TX_ITSELF | not applicable: a property of this transaction only | no observation of this txid | [0.0, None] |  |
| `dominant_peer_share` | E_network | no | TX_ITSELF | not applicable: a property of this transaction only | no observation of this txid | [0.0, 1.0] |  |
| `arrival_spread_seconds` | E_network | no | TX_ITSELF | not applicable: a property of this transaction only | no observation of this txid or no timestamps | [0.0, None] |  |
