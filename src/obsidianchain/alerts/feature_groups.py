"""Feature -> M0/M1/M2/M3, as a plain literal.

Generated from ``obsidianchain.features.dataset.ALL_FEATURE_GROUPS`` and
checked against it by ``tests/test_phase7_alerts.py``. It covers M4 as well,
so an artifact built with the optional mixing group can still be grouped -
an artifact built without it simply has no M4 columns to place. It exists as a literal because
``api/boundary.py`` forbids the API package from importing the feature
builders at all - and importing them just to read a column list would put
the whole recomputation path one attribute away from a request handler.

Standard library only. Regenerate with ``phase7-alerts --refresh-groups``
if the feature contract changes; the test fails until you do.
"""

from __future__ import annotations

GROUP_OF: dict[str, str] = {
    'n_txs_asof_t': 'M0',
    'n_txs_as_sender_asof_t': 'M0',
    'n_txs_as_receiver_asof_t': 'M0',
    'btc_sent_total_asof_t': 'M0',
    'btc_sent_mean_asof_t': 'M0',
    'btc_sent_max_asof_t': 'M0',
    'btc_received_total_asof_t': 'M0',
    'btc_received_mean_asof_t': 'M0',
    'btc_received_max_asof_t': 'M0',
    'fees_total_asof_t': 'M0',
    'fees_mean_asof_t': 'M0',
    'tx_size_mean_asof_t': 'M0',
    'input_fanin_mean_asof_t': 'M0',
    'output_fanout_mean_asof_t': 'M0',
    'active_timesteps_asof_t': 'M0',
    'timesteps_since_first_seen_asof_t': 'M0',
    'tx_per_active_timestep_asof_t': 'M0',
    'gap_mean_timesteps_asof_t': 'M0',
    'gap_max_timesteps_asof_t': 'M0',
    'sent_share_of_tx_asof_t': 'M0',
    'in_degree_asof_t': 'M1',
    'out_degree_asof_t': 'M1',
    'weighted_in_degree_asof_t': 'M1',
    'weighted_out_degree_asof_t': 'M1',
    'unique_counterparties_asof_t': 'M1',
    'counterparty_growth_rate_asof_t': 'M1',
    'local_clustering_coefficient_asof_t': 'M1',
    'cluster_size_asof_t': 'M1',
    'in_chain': 'M2',
    'chain_depth_max': 'M2',
    'position_in_chain': 'M2',
    'chain_count': 'M2',
    'chain_fanout_mean': 'M2',
    'hop_gap_median': 'M2',
    'value_retention_ratio': 'M2',
    'value_retention_available': 'M2',
    'net_pooled_observations': 'M3',
    'net_has_evidence': 'M3',
    'net_observer_coverage': 'M3',
    'net_observers_present': 'M3',
    'net_arrival_dispersion_mean': 'M3',
    'net_arrival_dispersion_max': 'M3',
    'net_reaches_production_minimum': 'M3',
    'mixing_tx_count_asof_t': 'M4',
    'mixing_tx_share_asof_t': 'M4',
    'mixing_score_max_asof_t': 'M4',
    'mixing_score_mean_asof_t': 'M4',
    'mixing_participants_max_asof_t': 'M4',
    'mixing_signal_available': 'M4',
}
