/**
 * One alert, opened.
 *
 * Layout follows the investigator's question order: what is this and how
 * confident is the model, then why, then the evidence, then the graph,
 * timeline, network context and provenance. Nothing on this page is computed
 * in the browser - every number came from the API.
 */
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { fetchAlert } from "../api/client";
import type { AlertDetail as AlertDetailData } from "../api/types";
import { ErrorState } from "./ErrorState";
import { EvidencePanel } from "./EvidencePanel";
import { InvestigationGraph } from "./InvestigationGraph";
import { CorrelationPanel } from "./CorrelationPanel";
import { NetworkContextPanel } from "./NetworkContextPanel";
import { ProvenancePanel } from "./ProvenancePanel";
import { Timeline } from "./Timeline";
import { WhyFlagged } from "./WhyFlagged";
import { Address, RiskBar, SeverityBadge, Skeleton, Value } from "./primitives";

export function AlertDetailPage() {
  const { alertId = "" } = useParams();
  const [data, setData] = useState<AlertDetailData | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [reloads, setReloads] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setError(null); setData(null);
    fetchAlert(alertId, controller.signal)
      .then((response) => { setData(response); setLoading(false); })
      .catch((cause) => {
        if ((cause as Error)?.name === "AbortError") return;
        setError(cause); setLoading(false);
      });
    return () => controller.abort();
  }, [alertId, reloads]);

  if (loading) {
    return <section className="panel"><Skeleton rows={10} /></section>;
  }
  if (error) {
    return (
      <>
        <p><Link to="/alerts">← Back to the alert queue</Link></p>
        <section className="panel">
          <ErrorState error={error} onRetry={() => setReloads((n) => n + 1)} />
        </section>
      </>
    );
  }
  if (!data) return null;

  const { summary, risk } = data;

  return (
    <>
      <p><Link to="/alerts">← Back to the alert queue</Link></p>

      <section className="panel">
        <div className="panel-head">
          <h2>Alert {summary.cluster_id}</h2>
          <SeverityBadge severity={summary.severity} />
          <span className="spacer" style={{ flex: 1 }} />
          <span className="mono small faint">{data.alert_id}</span>
        </div>
        <div className="panel-body">
          <div className="stat-row">
            <div className="stat">
              <span className="v" style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <RiskBar value={risk.score} severity={risk.severity} />
                <Value value={risk.score} digits={4} />
              </span>
              <span className="k">Calibrated risk ({risk.ranking_aggregation})</span>
            </div>
            <div className="stat">
              <span className="v"><Value value={risk.top_member_risk} digits={4} /></span>
              <span className="k">Highest member risk</span>
            </div>
            <div className="stat">
              <span className="v">{summary.members_scored.toLocaleString()}</span>
              <span className="k">Members scored</span>
            </div>
            <div className="stat">
              <span className="v">{summary.members_total.toLocaleString()}</span>
              <span className="k">Members in cluster</span>
            </div>
            <div className="stat">
              <span className="v">
                {summary.first_timestep === summary.last_timestep
                  ? `t${summary.first_timestep}`
                  : `t${summary.first_timestep}–${summary.last_timestep}`}
              </span>
              <span className="k">Active timesteps</span>
            </div>
            <div className="stat">
              <span className="v">#{summary.rank}</span>
              <span className="k">Queue rank</span>
            </div>
          </div>

          <h3 className="small muted" style={{ margin: "18px 0 6px" }}>
            All aggregations of member risk
          </h3>
          <div className="stat-row">
            {Object.entries(risk.aggregations).map(([name, value]) => (
              <div className="stat" key={name}>
                <span className="v" style={{ fontSize: 15 }}><Value value={value} digits={4} /></span>
                <span className="k">
                  {name}{name === risk.ranking_aggregation ? " · ranking" : ""}
                </span>
              </div>
            ))}
          </div>

          <p className="note">{data.meaning}</p>
          <p className="note">{data.score_scope}</p>
        </div>
      </section>

      <div className="grid-2">
        <div>
          <WhyFlagged data={data.why_flagged} />
          <EvidencePanel evidence={data.evidence} />
          <InvestigationGraph alert={data} />
          <CorrelationPanel correlation={data.correlation} />
        </div>
        <div>
          <MembersPanel data={data} />
          <NetworkContextPanel context={data.network_context} />
          <Timeline timeline={data.timeline} observedAt={summary.last_timestep} />
          <ProvenancePanel provenance={data.provenance} />
        </div>
      </div>
    </>
  );
}

function MembersPanel({ data }: { data: AlertDetailData }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Member addresses</h2>
        <span className="small muted">
          {data.members.shown} of {data.members.total_scored.toLocaleString()} shown
        </span>
      </div>
      <div className="panel-body flush" style={{ maxHeight: 340, overflowY: "auto" }}>
        <table>
          <thead>
            <tr><th>Address</th><th>Severity</th><th className="num">Risk</th>
              <th className="num">As-of</th></tr>
          </thead>
          <tbody>
            {data.members.rows.map((m) => (
              <tr key={m.address}>
                <td><Address value={m.address} /></td>
                <td><SeverityBadge severity={m.severity} /></td>
                <td className="num"><Value value={m.risk_score} digits={4} /></td>
                <td className="num mono">t{m.observed_at_timestep}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data.members.withheld > 0 ? (
        <div className="panel-body">
          <p className="note" style={{ marginTop: 0 }}>
            {data.members.withheld.toLocaleString()} further scored members are not
            listed. The highest-risk members are shown first.
          </p>
        </div>
      ) : null}
    </section>
  );
}
