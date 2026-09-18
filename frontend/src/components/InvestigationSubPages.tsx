/**
 * Investigation sub-pages: Graph, Timeline, Network, Evidence.
 *
 * What changed
 * ------------
 * These four pages used to call `fetchAlerts({limit: 50})` and render the
 * GLOBAL top fifty regardless of which case the URL named - so a case that
 * had referenced nothing still showed a populated workspace, and a case id
 * that did not exist showed one too.
 *
 * Now each page draws from the alerts THIS investigation has referenced. A
 * case with no references says so, which is the honest answer: these are
 * views onto the case's own material, not onto the artifact's.
 */
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { fetchAlert } from "../api/client";
import * as api from "../api/console";
import type { AlertDetail, CaseAlertRow } from "../api/types";
import { Skeleton } from "./primitives";
import { DispositionBadge } from "./CaseChrome";
import { InvestigationGraph } from "./InvestigationGraph";
import { Timeline } from "./Timeline";
import { CorrelationPanel } from "./CorrelationPanel";
import { NetworkContextPanel } from "./NetworkContextPanel";
import { EvidencePanel } from "./EvidencePanel";
import { SeparationEvidencePanel } from "./SeparationEvidencePanel";

function AlertSelector({
  rows, selected, onSelect,
}: {
  rows: CaseAlertRow[];
  selected: string;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="alert-selector">
      <label className="filter-label" htmlFor="case-alert-pick">
        Referenced alert
      </label>
      <select id="case-alert-pick" value={selected}
        onChange={(e) => onSelect(e.target.value)}>
        {rows.map((r) => (
          <option key={r.alert_id} value={r.alert_id}>
            {r.alert_id}
            {r.disposition ? ` · ${r.disposition.state}` : ""}
            {r.stale === true ? " · STALE" : ""}
          </option>
        ))}
      </select>
      {rows.find((r) => r.alert_id === selected)?.disposition && (
        <DispositionBadge
          state={rows.find((r) => r.alert_id === selected)?.disposition?.state}
        />
      )}
    </div>
  );
}

/**
 * Loads the case's referenced alerts, then the selected one's analytical detail.
 *
 * `stale` is carried through so a page can refuse to draw rather than fetch
 * an alert id the current artifact would reject with a 409.
 */
function useCaseAlerts() {
  const { invId = "" } = useParams();
  const [rows, setRows] = useState<CaseAlertRow[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [detail, setDetail] = useState<AlertDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    api.listCaseAlerts(invId, controller.signal)
      .then((r) => {
        setRows(r.alerts);
        const first = r.alerts.find((a) => a.stale !== true) ?? r.alerts[0];
        if (first) setSelectedId(first.alert_id);
        else setLoading(false);
      })
      .catch(() => setLoading(false));
    return () => controller.abort();
  }, [invId]);

  useEffect(() => {
    if (!selectedId) return;
    const row = rows.find((r) => r.alert_id === selectedId);
    if (row?.stale === true) { setDetail(null); setLoading(false); return; }
    setLoading(true);
    fetchAlert(selectedId)
      .then(setDetail)
      .catch(() => setDetail(null))
      .finally(() => setLoading(false));
  }, [selectedId, rows]);

  const selected = rows.find((r) => r.alert_id === selectedId);
  return { invId, rows, selectedId, setSelectedId, detail, loading, selected };
}

function SubPage({
  title,
  children,
}: {
  title: string;
  children: (ctx: ReturnType<typeof useCaseAlerts>) => React.ReactNode;
}) {
  const ctx = useCaseAlerts();
  return (
    <>
      <div className="page-header"><h1>{title}</h1></div>
      {ctx.rows.length === 0 ? (
        <section className="panel">
          <div className="panel-body">
            <p className="muted" style={{ marginTop: 0 }}>
              This investigation has not referenced any alerts, so there is
              nothing to draw here. Add alerts from the{" "}
              <Link to={`/inv/${ctx.invId}/alerts`}>alerts page</Link>.
            </p>
            <p className="note">
              This page shows only the case's own material. It does not fall
              back to the analytical run's top-ranked alerts, which belong to
              the pipeline rather than to this investigation.
            </p>
          </div>
        </section>
      ) : (
        <>
          <AlertSelector rows={ctx.rows} selected={ctx.selectedId}
            onSelect={ctx.setSelectedId} />
          {ctx.selected?.stale === true ? (
            <section className="panel">
              <div className="panel-body">
                <div className="banner banner-error" style={{ marginTop: 0 }}>
                  <h4>This reference is stale</h4>
                  <p>
                    It was referenced under run{" "}
                    <code>{ctx.selected.run_fingerprint}</code>, which is not
                    the artifact currently on disk. Analytical views cannot be
                    resolved for it. The reference and its decision history are
                    kept.
                  </p>
                </div>
              </div>
            </section>
          ) : ctx.loading ? (
            <Skeleton rows={8} />
          ) : (
            children(ctx)
          )}
        </>
      )}
    </>
  );
}

export function GraphSubPage() {
  return (
    <SubPage title="Investigation graph">
      {({ detail }) =>
        detail ? (
          <div style={{ minHeight: 500 }}><InvestigationGraph alert={detail} /></div>
        ) : <p className="muted">No graph data available.</p>
      }
    </SubPage>
  );
}

export function TimelineSubPage() {
  return (
    <SubPage title="Activity timeline">
      {({ detail }) =>
        detail ? (
          <Timeline timeline={detail.timeline}
            observedAt={detail.summary.last_timestep} />
        ) : <p className="muted">No timeline data available.</p>
      }
    </SubPage>
  );
}

export function NetworkSubPage() {
  return (
    <SubPage title="Network intelligence">
      {({ detail }) =>
        detail ? (
          <>
            <NetworkContextPanel context={detail.network_context} />
            {detail.correlation.transactions.length > 0 && (
              <CorrelationPanel correlation={detail.correlation} />
            )}
          </>
        ) : <p className="muted">No network data available.</p>
      }
    </SubPage>
  );
}

/**
 * Evidence, in the two senses the backend actually distinguishes.
 *
 * M0-M3 feature values describe the ledger and the model's inputs. The
 * separation funnel describes whether the network layer could tell two
 * candidate components apart. They are different evidence and appear as
 * different panels rather than under one word.
 */
export function EvidenceSubPage() {
  return (
    <SubPage title="Evidence">
      {({ detail, selectedId }) =>
        detail ? (
          <>
            <EvidencePanel evidence={detail.evidence} />
            <SeparationEvidencePanel alertId={selectedId} />
          </>
        ) : <p className="muted">No evidence data available.</p>
      }
    </SubPage>
  );
}
