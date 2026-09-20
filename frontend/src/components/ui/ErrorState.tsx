/**
 * The backend's error taxonomy, preserved.
 *
 * A stale run fingerprint gets its own treatment because it means something
 * specific and actionable: the artifact was regenerated under the
 * investigator, so the alert they are holding may no longer refer to the
 * same cluster. Flattening it into "something went wrong" would hide the one
 * error this system was built to make visible.
 */
import { ApiError } from "../../api/client";

export function ErrorState({ error, onRetry }:
  { error: unknown; onRetry?: () => void }) {
  const api = error instanceof ApiError ? error : null;

  if (api?.isStale) {
    return (
      <div className="state">
        <div className="banner banner-stale" style={{ textAlign: "left" }}>
          <h4>Run fingerprint is stale</h4>
          <p>
            This alert was opened against a different run of the pipeline. The
            artifacts have since been regenerated, and a cluster id does not
            necessarily refer to the same cluster across runs — so this alert
            is not shown rather than shown against the wrong data.
          </p>
          <p style={{ marginTop: 8 }}>Return to the queue and re-open it from the current list.</p>
          <span className="mono small faint" style={{ display: "block", marginTop: 10 }}>
            {api.detail}
          </span>
        </div>
      </div>
    );
  }

  const title = api?.isRefusal
    ? "The API refused to serve this artifact"
    : api?.kind === "alert_not_found"
      ? "No such alert in the current run"
      : api?.kind === "network"
        ? "Cannot reach the API"
        : "Request failed";

  return (
    <div className="state">
      <div className="banner banner-error" style={{ textAlign: "left" }}>
        <h4>{title}</h4>
        <p>{api?.detail ?? String((error as Error)?.message ?? error)}</p>
      </div>
      {onRetry ? <button onClick={onRetry}>Retry</button> : null}
    </div>
  );
}
