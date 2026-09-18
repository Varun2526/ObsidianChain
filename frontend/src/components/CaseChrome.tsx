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
import type { DispositionState, RunStatus, Disposition } from "../api/types";

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
