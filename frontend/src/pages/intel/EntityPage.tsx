/**
 * Address intelligence: what the ledger shows, who else asserts something
 * about it, and what the model associated with it - in that order, and each
 * in its own section so none is read as another.
 */
import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { getAddress, traceGraph } from "../../api/intel";
import type { AddressProfile, Counterparty } from "../../api/intel";
import { FlowPreview } from "../../components/graph/FlowPreview";
import { StepBars } from "../../components/ui/charts";
import { ErrorState } from "../../components/ui/ErrorState";
import { Icon } from "../../components/ui/Icon";
import {
  AddressLink, AnnotationChips, CopyButton, EvidenceTag, Metric, PageHeader, SevTag, TxLink, btc, int, pct,
} from "../../components/ui/intel";
import { Skeleton } from "../../components/ui/primitives";
import { useApi } from "../../lib/useApi";

export function EntityPage() {
  const { address = "" } = useParams();
  const profile = useApi((s) => getAddress(address, s), [address]);
  const flow = useApi((s) => traceGraph({ addresses: [address], direction: "both", hops: 1, maxNodes: 160 }, s), [address]);

  if (profile.loading && !profile.data) {
    return <><PageHeader eyebrow="Address" title={<span className="mono">{address}</span>} /><div className="panel"><Skeleton rows={6} /></div></>;
  }
  if (profile.error) {
    return <><PageHeader eyebrow="Address" title={<span className="mono">{address}</span>} /><ErrorState error={profile.error} onRetry={profile.reload} /></>;
  }
  const p = profile.data!;
  const explorer = `/graph?address=${encodeURIComponent(address)}&direction=both&hops=2`;

  return (
    <>
      <PageHeader
        eyebrow={<><Link to="/graph">Intelligence</Link><span>/</span><span>Address</span></>}
        title={<span className="mono" style={{ fontSize: 18, overflowWrap: "anywhere" }}>{p.address}</span>}
        sub={<AnnotationChips model={p.model.alerts[0] ? { ...p.model.alerts[0], alerts: p.model.alerts.length, scope: p.model.scope } : null}
                              cluster={p.cluster} watchlist={p.watchlist} />}
        actions={<>
          <CopyButton value={p.address} label="Copy address" />
          <Link className="btn btn-sm btn-primary" to={explorer}><Icon name="graph" size={14} />Trace in explorer</Link>
        </>}
      />

      {p.watchlist.length > 0 && (
        <div className="banner banner-warn" role="note">
          <h4>External attribution</h4>
          <p>
            {p.watchlist.map((w) => `${w.source.replace("WATCHLIST:", "")}${w.label ? ` (${w.label})` : ""}`).join("; ")}.
            {" "}Asserted by the named source, not by this system. Verify against the source before relying on it.
          </p>
        </div>
      )}

      <div className="section-title"><EvidenceTag kind="chain" /> Observed activity</div>
      <div className="metric-strip">
        <Metric k="Transactions" v={int(p.observed.transactions)} d={`${int(p.observed.as_input)} spending · ${int(p.observed.as_output)} receiving`} />
        <Metric k="Active" v={p.observed.first_timestep == null ? "n/a" : `t${p.observed.first_timestep}–t${p.observed.last_timestep}`} d={`${p.observed.active_timesteps} active timesteps`} />
        <Metric k="Paid by" v={int(p.counterparties.paid_by_total)} d="distinct funding addresses" />
        <Metric k="Paid to" v={int(p.counterparties.paid_to_total)} d="distinct destination addresses" />
        <Metric k="Cluster" v={p.cluster ? int(p.cluster.size) : "none"} d={p.cluster ? `heuristic cluster ${p.cluster.cluster_id}` : "not clustered"} />
      </div>

      <div className="grid-main-side">
        <div className="stack">
          <section className="panel">
            <div className="panel-head"><h2>Activity by timestep</h2><span className="small faint">transactions per Elliptic++ timestep (about two weeks each)</span></div>
            <div className="panel-body">
              <StepBars
                ariaLabel={`Transactions per timestep for ${p.address}`}
                data={p.observed.timeline.map((t) => ({ x: t.timestep, values: [t.as_input, t.as_output] }))}
                colors={["var(--oc-sev-high)", "var(--oc-ev-chain)"]}
                labels={["spent from this address", "received by this address"]}
              />
            </div>
          </section>

          {flow.data && flow.data.graph.node_count > 0 && (
            <FlowPreview
              nodes={flow.data.graph.nodes}
              edges={flow.data.graph.edges}
              truncated={flow.data.truncated}
              explorerHref={explorer}
              title="Money flow, one hop each way"
              note={<>Value moves left to right in the flow layout. {flow.data.hub_transactions_skipped.length > 0 && `${flow.data.hub_transactions_skipped.length} transaction(s) with more than ${flow.data.hub_threshold} participants were not crossed. `}Sharing a transaction is not common ownership.</>}
            />
          )}
          {flow.error != null && <ErrorState error={flow.error} onRetry={flow.reload} />}

          <TransactionsTable p={p} />
        </div>

        <div className="stack">
          <CounterpartyPanel title="Paid by" hint="addresses that funded transactions paying this address" rows={p.counterparties.paid_by} total={p.counterparties.paid_by_total} />
          <CounterpartyPanel title="Paid to" hint="addresses paid by transactions this address funded" rows={p.counterparties.paid_to} total={p.counterparties.paid_to_total} />
          {p.counterparties.hub_transactions_skipped > 0 && (
            <p className="note">{p.counterparties.hub_transactions_skipped} transaction(s) with more than {p.counterparties.hub_threshold} participants are excluded from counterparties: at that size they are exchange batches or mixers, and every participant would look related.</p>
          )}
        </div>
      </div>

      <ModelSection p={p} />

      <div className="section-title"><EvidenceTag kind="network" /> Relay observations</div>
      <section className="panel">
        <div className="panel-body">
          {p.network.observations.length === 0 ? (
            <p className="muted small">No announcement observations for this address's transactions.</p>
          ) : (
            <table>
              <thead><tr><th>Transaction</th><th className="num">Announcing peers</th><th className="num">ASNs</th></tr></thead>
              <tbody>{p.network.observations.map((o) => (
                <tr key={o.txid}><td><TxLink txid={o.txid} /></td><td className="num">{o.peers}</td><td className="num">{o.asns}</td></tr>
              ))}</tbody>
            </table>
          )}
          <p className="note" style={{ marginTop: 8 }}>{p.network.meaning}</p>
        </div>
      </section>

      <p className="note">
        Chain index <span className="mono">{p.provenance.chain_index_fingerprint}</span>. {p.provenance.chain_scope} Class labels are never served.
      </p>
    </>
  );
}

function CounterpartyPanel({ title, hint, rows, total }: { title: string; hint: string; rows: Counterparty[]; total: number }) {
  return (
    <section className="panel">
      <div className="panel-head"><h2>{title}</h2><span className="small faint">{total > rows.length ? `top ${rows.length} of ${int(total)}` : int(total)}</span></div>
      <div className="panel-body flush">
        {rows.length === 0 ? <p className="muted small" style={{ padding: 16 }}>None observed.</p> : (
          <table>
            <thead><tr><th>Address</th><th className="num">Shared tx</th><th>Notes</th></tr></thead>
            <tbody>{rows.map((c) => (
              <tr key={c.address}>
                <td><AddressLink address={c.address} /></td>
                <td className="num">{c.shared_transactions}</td>
                <td><AnnotationChips model={c.model} watchlist={c.watchlist} compact /></td>
              </tr>
            ))}</tbody>
          </table>
        )}
      </div>
      <div className="panel-foot">{hint}</div>
    </section>
  );
}

function TransactionsTable({ p }: { p: AddressProfile }) {
  const [role, setRole] = useState<"all" | "input" | "output">("all");
  const rows = useMemo(() => p.observed.transactions_list.filter((t) => role === "all" || t.role === role || t.role === "both"),
    [p, role]);
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Transactions</h2>
        <span className="small faint">{p.observed.transactions_shown < p.observed.transactions ? `latest ${p.observed.transactions_shown} of ${int(p.observed.transactions)}` : int(p.observed.transactions)}</span>
        <span className="spacer" />
        <div className="btn-group" role="group" aria-label="Role filter">
          {(["all", "input", "output"] as const).map((r) => (
            <button key={r} type="button" className="btn btn-sm" aria-pressed={role === r} onClick={() => setRole(r)}>
              {r === "all" ? "All" : r === "input" ? "Spent" : "Received"}
            </button>
          ))}
        </div>
      </div>
      <div className="panel-body flush table-wrap" style={{ maxHeight: 420 }}>
        <table>
          <thead><tr><th>Txid</th><th className="num">Step</th><th>Role</th><th className="num">In · out</th><th className="num">Value out</th><th className="num">Fee</th></tr></thead>
          <tbody>{rows.map((t) => (
            <tr key={t.txid}>
              <td><TxLink txid={t.txid} /></td>
              <td className="num">{t.timestep ?? "n/a"}</td>
              <td className="small">{t.role === "input" ? "spent" : t.role === "output" ? "received" : "both"}</td>
              <td className="num">{t.n_inputs} · {t.n_outputs}</td>
              <td className="num">{btc(t.out_btc)}</td>
              <td className="num">{btc(t.fee_btc)}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <div className="panel-foot">Per-transaction totals. The source has no per-address amounts, so none are shown.</div>
    </section>
  );
}

function ModelSection({ p }: { p: AddressProfile }) {
  const top = p.model.explanations;
  const max = Math.max(1e-9, ...top.map((e) => Math.abs(e.contribution)));
  return (
    <>
      <div className="section-title"><EvidenceTag kind="model" /> Model association</div>
      <section className="panel">
        <div className="panel-body">
          {p.model.alerts.length === 0 ? (
            <p className="muted small">This address is not a member of any alert in the Phase 7 reference run. That is not a clearance: the run scores only addresses in scope.</p>
          ) : (
            <>
              <div className="banner banner-model" style={{ marginBottom: 12 }}>
                <h4>A learned association, not proof</h4>
                <p>{p.model.meaning}</p>
              </div>
              <table style={{ marginBottom: 16 }}>
                <thead><tr><th>Alert</th><th>Severity</th><th className="num">Model risk</th><th className="num">Scored at step</th></tr></thead>
                <tbody>{p.model.alerts.map((a) => (
                  <tr key={a.alert_id}>
                    <td><Link className="mono row-link" to={`/alerts/${encodeURIComponent(a.alert_id)}`}>cluster {a.alert_id.split(":")[1]}</Link></td>
                    <td><SevTag severity={a.severity} /></td>
                    <td className="num">{pct(a.risk_score, 2)}</td>
                    <td className="num">{a.observed_at_timestep ?? "n/a"}</td>
                  </tr>
                ))}</tbody>
              </table>
              {top.length > 0 && (
                <>
                  <h3 style={{ marginBottom: 8 }}>What moved the score <span className="small faint">(SHAP contribution, log-odds)</span></h3>
                  {top.map((e) => (
                    <div className="contrib" key={`${e.alert_id}-${e.feature}`}>
                      <span className="contrib-label" title={e.feature}>{e.feature}</span>
                      <span className="contrib-bar" aria-hidden="true">
                        <span className={`fill ${e.contribution >= 0 ? "pos" : "neg"}`} style={{ width: `${(Math.abs(e.contribution) / max) * 50}%` }} />
                      </span>
                      <span className="num small" style={{ textAlign: "right" }}>{e.contribution >= 0 ? "+" : ""}{e.contribution.toFixed(3)}</span>
                    </div>
                  ))}
                  <p className="note" style={{ marginTop: 8 }}>Positive values raised the model's score, negative values lowered it. A contribution describes the model, not the address.</p>
                </>
              )}
            </>
          )}
        </div>
      </section>
    </>
  );
}
