/**
 * One transaction: who funded it, who it paid, what the mixing scan says
 * about its shape, and which peers relayed it. Per-address amounts are not
 * shown because the source does not have them.
 */
import { Fragment } from "react";
import { Link, useParams } from "react-router-dom";

import { getTransactionDrilldown } from "../../api/console";
import { traceGraph } from "../../api/intel";
import type { TxParty } from "../../api/types";
import { FlowPreview } from "../../components/graph/FlowPreview";
import { ErrorState } from "../../components/ui/ErrorState";
import { Icon } from "../../components/ui/Icon";
import { AddressLink, AnnotationChips, EvidenceTag, Metric, PageHeader, btc, fixed, int } from "../../components/ui/intel";
import { Skeleton } from "../../components/ui/primitives";
import { useApi } from "../../lib/useApi";

export function TransactionPage() {
  const { txid = "" } = useParams();
  const tx = useApi((s) => getTransactionDrilldown(txid, s), [txid]);
  const flow = useApi((s) => traceGraph({ txids: [Number(txid)], direction: "both", hops: 1, maxNodes: 160 }, s), [txid]);

  const header = <PageHeader eyebrow={<><Link to="/graph">Intelligence</Link><span>/</span><span>Transaction</span></>} title={<span className="mono">{txid}</span>} />;
  if (tx.loading && !tx.data) return <>{header}<div className="panel"><Skeleton rows={6} /></div></>;
  if (tx.error) return <>{header}<ErrorState error={tx.error} onRetry={tx.reload} /></>;
  const t = tx.data!;
  const explorer = `/graph?txid=${t.txid}&direction=both&hops=2`;

  return (
    <>
      <PageHeader
        eyebrow={<><Link to="/graph">Intelligence</Link><span>/</span><span>Transaction</span></>}
        title={<span className="mono">{t.txid}</span>}
        sub={t.timestep != null ? `Elliptic++ timestep ${t.timestep}` : undefined}
        actions={<Link className="btn btn-sm btn-primary" to={explorer}><Icon name="graph" size={14} />Trace in explorer</Link>}
      />

      <div className="section-title"><EvidenceTag kind="chain" /> Observed</div>
      <div className="metric-strip">
        <Metric k="Inputs" v={int(t.n_inputs ?? t.input_count)} d="funding addresses" />
        <Metric k="Outputs" v={int(t.n_outputs ?? t.output_count)} d="paid addresses" />
        <Metric k="Value in" v={t.in_btc == null ? "n/a" : (t.in_btc).toLocaleString(undefined, { maximumFractionDigits: 6 })} d="BTC" />
        <Metric k="Value out" v={t.out_btc == null ? "n/a" : (t.out_btc).toLocaleString(undefined, { maximumFractionDigits: 6 })} d="BTC" />
        <Metric k="Fee" v={t.fee_btc == null ? "n/a" : t.fee_btc.toLocaleString(undefined, { maximumFractionDigits: 8 })} d="BTC" />
      </div>

      <section className="panel">
        <div className="panel-head"><h2>Flow</h2><span className="small faint">funding addresses, the transaction, paid addresses</span></div>
        <div className="panel-body">
          <div className="flow">
            <PartyList title="Funded by" rows={t.inputs} />
            <div className="flow-hub">
              <span className="k">Transaction</span>
              <span className="v">{btc(t.out_btc ?? null)}</span>
              <span className="k">out · fee {btc(t.fee_btc ?? null)}</span>
            </div>
            <PartyList title="Paid" rows={t.outputs} />
          </div>
        </div>
      </section>

      {flow.data && flow.data.graph.node_count > 0 && (
        <FlowPreview nodes={flow.data.graph.nodes} edges={flow.data.graph.edges} truncated={flow.data.truncated}
                     explorerHref={explorer} title="Money flow, one hop each way" />
      )}

      <div className="grid-2">
        <section className="panel">
          <div className="panel-head"><h2>Mixing structure</h2><EvidenceTag kind="rule">Heuristic scan</EvidenceTag></div>
          <div className="panel-body">
            {t.mixing.available ? (
              <>
                <dl className="kv">
                  <dt>Class</dt><dd><span className="chip">{t.mixing.mixing_class}</span></dd>
                  <dt>Score</dt><dd className="num">{fixed(t.mixing.mixing_score ?? null)}</dd>
                  {Object.entries(t.mixing.signals ?? {}).map(([k, v]) => (
                    <Fragment key={k}><dt>{k.replace(/_/g, " ")}</dt><dd className="num">{fixed(v)}</dd></Fragment>
                  ))}
                </dl>
                <p className="note" style={{ marginTop: 8 }}>{t.mixing.meaning}</p>
              </>
            ) : <p className="muted small">{t.mixing.meaning}</p>}
          </div>
        </section>
        <section className="panel">
          <div className="panel-head"><h2>Relay peers</h2><EvidenceTag kind="network" /></div>
          <div className="panel-body flush">
            {t.announcing_peers.length === 0 ? <p className="muted small" style={{ padding: 16 }}>No announcement observations for this transaction.</p> : (
              <table>
                <thead><tr><th>Peer</th><th className="num">ASN</th><th className="num">Observers</th></tr></thead>
                <tbody>{t.announcing_peers.map((p) => (
                  <tr key={`${p.ip}:${p.port}`}><td className="mono">{p.ip}{p.port ? `:${p.port}` : ""}</td><td className="num">{p.asn ?? "n/a"}</td><td className="num">{p.observers}</td></tr>
                ))}</tbody>
              </table>
            )}
          </div>
          <div className="panel-foot">{t.limitation}</div>
        </section>
      </div>

      {t.associated_clusters.length > 0 && (
        <p className="note">Alert clusters touching this transaction: {t.associated_clusters.join(", ")}.</p>
      )}
      {t.provenance && <p className="note">Chain index <span className="mono">{t.provenance.chain_index_fingerprint}</span>. No class labels are served.</p>}
    </>
  );
}

function PartyList({ title, rows }: { title: string; rows: TxParty[] }) {
  return (
    <div>
      <h3 className="field-label" style={{ marginBottom: 6 }}>{title} ({rows.length})</h3>
      <ul className="flow-list">
        {rows.slice(0, 60).map((r) => (
          <li key={r.address}>
            <AddressLink address={r.address} />
            <span className="spacer" />
            <AnnotationChips model={r.model} watchlist={r.watchlist} compact />
          </li>
        ))}
        {rows.length > 60 && <li className="faint small">{rows.length - 60} more not listed</li>}
      </ul>
    </div>
  );
}
