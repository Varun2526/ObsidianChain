/**
 * Small shared pieces that carry the domain distinctions the UI must not blur.
 *
 * Three of them exist because the same word was doing two jobs:
 *
 *   `SeverityBadge`      the MODEL's assessment (elsewhere, in primitives)
 *   `DispositionBadge`   the INVESTIGATOR's decision
 *   `RunStatusChip`      whether a case's analytical binding still matches disk
 *
 * They are visually distinct on purpose. A model severity and an investigator
 * disposition are different kinds of claim, and rendering them in the same
 * shape is how a ranking signal starts reading as a finding.
 */
import { NavLink } from "react-router-dom";
import type { DispositionState, RunStatus, Disposition, Investigation } from "../../api/types";

const DISPOSITION_TONE: Record<DispositionState, string> = {
  NEW: "new",
  TRIAGED: "triaged",
  IN_REVIEW: "review",
  CONFIRMED: "confirmed",
  DISMISSED: "dismissed",
  ESCALATED: "escalated",
};

export function DispositionBadge({
  state,
}: {
  state: DispositionState | null | undefined;
}) {
  if (!state) return <span className="faint small">no decision</span>;
  return (
    <span className={`disp disp-${DISPOSITION_TONE[state]}`}>
      {state.replace("_", " ")}
    </span>
  );
}

const RUN_LABEL: Record<RunStatus, string> = {
  UNBOUND: "no run bound",
  CURRENT: "current",
  STALE: "STALE",
  UNVERIFIABLE: "unverifiable",
};

export function RunStatusChip({ status }: { status: RunStatus }) {
  return (
    <span className={`runchip runchip-${status.toLowerCase()}`}>
      {RUN_LABEL[status]}
    </span>
  );
}

/**
 * The stale-run banner.
 *
 * Shown instead of, never in place of, the data it describes: a stale
 * reference is kept and labelled, because removing it would erase an
 * investigator's decision and re-pointing it would attach that decision to a
 * cluster they never saw.
 */
export function StaleRunBanner({
  bound,
  current,
  count,
}: {
  bound: string | null;
  current: string | null;
  count?: number;
}) {
  return (
    <div className="banner banner-error">
      <h4>Analytical run changed — referenced results are stale</h4>
      <p>
        This investigation is bound to run <code>{bound ?? "—"}</code> and the
        artifact currently on disk is run <code>{current ?? "unavailable"}</code>.
        {typeof count === "number" && count > 0
          ? ` ${count} referenced alert${count === 1 ? "" : "s"} cannot be resolved against the current run.`
          : ""}{" "}
        Nothing has been removed or re-pointed: alert ids do not survive a
        regeneration, so a re-point would silently attach recorded decisions to
        different clusters.
      </p>
    </div>
  );
}

export function DispositionHistory({ history }: { history: Disposition[] }) {
  if (!history.length) {
    return <p className="muted small">No decision has been recorded yet.</p>;
  }
  return (
    <ol className="disp-history">
      {history.map((d) => (
        <li key={d.id} className={d.active ? "active" : "superseded"}>
          <div className="disp-history-head">
            <DispositionBadge state={d.state} />
            {!d.active && <span className="faint small">superseded</span>}
            <span className="spacer" />
            <span className="small muted">
              {d.decided_by_display_name ?? d.decided_by_username ?? d.decided_by}
              {" · "}
              {new Date(d.decided_at).toLocaleString()}
            </span>
          </div>
          {d.rationale && <p className="disp-rationale">{d.rationale}</p>}
        </li>
      ))}
    </ol>
  );
}

const INV_NAV_TABS = [
  { sub: "", label: "OVERVIEW" },
  { sub: "/alerts", label: "ALERTS" },
  { sub: "/graph", label: "GRAPH" },
  { sub: "/timeline", label: "TIMELINE" },
  { sub: "/network", label: "NETWORK" },
  { sub: "/evidence", label: "EVIDENCE" },
  { sub: "/notes", label: "NOTES" },
  { sub: "/report", label: "REPORT" },
  { sub: "/review", label: "REVIEW" },
  { sub: "/history", label: "AUDIT LOG" },
];

export function PersistentCaseHeader({ inv }: { inv: Investigation }) {
  const dataset = inv.datasets?.[0];
  const runStatus = inv.run_status ?? inv.analytical_run?.status ?? "UNBOUND";
  const runFp = inv.bound_run_fingerprint
    ? inv.bound_run_fingerprint.slice(0, 16)
    : "UNBOUND";

  const totalAlerts = inv.summary?.alerts_referenced ?? 0;
  const outstanding = inv.summary?.outstanding ?? 0;
  const escalated = inv.summary?.dispositions_by_state?.ESCALATED ?? 0;
  const confirmed = inv.summary?.dispositions_by_state?.CONFIRMED ?? 0;

  return (
    <div className="case-persistent-header no-print">
      <div className="case-header-top">
        <div className="case-header-title-block">
          <span className="case-header-label">{inv.case_label}</span>
          <h2 className="case-header-title">{inv.name}</h2>
          <span className={`status-badge status-${inv.status.toLowerCase()}`}>
            ● {inv.status}
          </span>
          <div style={{ display: "flex", gap: 10, alignItems: "center", marginLeft: 8, fontSize: 13 }}>
            <span className="mono"><strong>{totalAlerts}</strong> Alerts</span>
            {outstanding > 0 && (
              <>
                <span className="faint">·</span>
                <span className="mono" style={{ color: "var(--high)" }}><strong>{outstanding}</strong> Outstanding</span>
              </>
            )}
            {escalated > 0 && (
              <>
                <span className="faint">·</span>
                <span className="mono" style={{ color: "var(--critical)" }}><strong>{escalated}</strong> Escalated</span>
              </>
            )}
            {confirmed > 0 && (
              <>
                <span className="faint">·</span>
                <span className="mono" style={{ color: "#10b981" }}><strong>{confirmed}</strong> Confirmed</span>
              </>
            )}
          </div>
        </div>
        <div className="case-header-context">
          <div className="case-ctx-item">
            <span className="case-ctx-k">Owner</span>
            <span className="case-ctx-v">
              {inv.owner?.display_name ?? inv.owner?.username ?? "Unassigned"}
            </span>
          </div>
          <div className="case-ctx-item">
            <span className="case-ctx-k">Dataset</span>
            <span className="case-ctx-v mono" title={dataset?.filename}>
              {dataset ? dataset.filename : "None"}
            </span>
          </div>
          <div className="case-ctx-item">
            <span className="case-ctx-k">Analysis</span>
            <span className="case-ctx-v">
              <span className={`status-badge status-${String(dataset?.analysis_run?.status ?? "NOT_RUN").toLowerCase()}`}>
                {dataset?.analysis_run?.status ?? "NOT_RUN"}
              </span>
            </span>
          </div>
          <div className="case-ctx-item">
            <span className="case-ctx-k">Run</span>
            <span className="case-ctx-v mono">
              {runFp}
              {" "}
              <RunStatusChip status={runStatus as RunStatus} />
            </span>
          </div>
        </div>
      </div>

      <nav className="case-tabs" aria-label="Investigation Navigation">
        {INV_NAV_TABS.map((tab) => {
          const path = `/inv/${inv.id}${tab.sub}`;
          return (
            <NavLink
              key={tab.sub}
              to={path}
              end={tab.sub === ""}
              className={({ isActive }) => `case-tab${isActive ? " active" : ""}`}
            >
              {tab.label}
            </NavLink>
          );
        })}
      </nav>
    </div>
  );
}
