/**
 * The only data source this dashboard has.
 *
 * There is no second path to the numbers: no local scoring, no cached
 * heuristics, no derived statistic computed in the browser. If the API does
 * not return it, the UI does not show it.
 *
 * Errors are preserved as the backend's own taxonomy rather than flattened
 * to "something went wrong". A stale run fingerprint in particular has to
 * reach the investigator intact - it means the artifact was regenerated under
 * them and the alert they are looking at no longer refers to the same
 * cluster, which is exactly the silent re-pointing the fingerprint exists to
 * prevent.
 *
 * These routes require a session. They did not always: the alerts, the risk
 * scores, the SHAP explanations and the separation evidence were readable
 * with one unauthenticated GET, which meant the console's login, its roles
 * and its case ownership could all be stepped around. Every request
 * therefore carries the session cookie, and a 401 is reported once to the
 * whole application rather than handled differently in each caller.
 */

import type {
  AlertDetail,
  AlertFilters,
  AlertListResponse,
  ApiErrorKind,
  IngestResult,
} from "./types";

const BASE = "/api";

/**
 * One place the application learns that its session has ended.
 *
 * Any call may be the one that discovers it - a background refresh as
 * readily as a click - so the discovery is broadcast instead of being
 * handled locally. `AuthProvider` subscribes and clears the identity, which
 * sends the router to /login. Without this a lapsed session leaves the shell
 * drawn and every panel failing individually, which reads as the application
 * being broken rather than as having been signed out.
 *
 * Only 401 fires it. A 403 means the session is perfectly valid and this
 * user may not have that thing - logging them out over it would be wrong.
 */
type SessionListener = () => void;
const sessionListeners = new Set<SessionListener>();

export function onSessionLost(listener: SessionListener): () => void {
  sessionListeners.add(listener);
  return () => sessionListeners.delete(listener);
}

export function reportUnauthenticated(status: number): void {
  if (status !== 401) return;
  for (const listener of [...sessionListeners]) {
    try {
      listener();
    } catch {
      // A misbehaving subscriber must not stop the others from learning.
    }
  }
}

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number;
  readonly detail: string;

  constructor(kind: ApiErrorKind, status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
    this.detail = detail;
  }

  /** Whether the artifact moved under the caller. Rendered distinctly. */
  get isStale(): boolean {
    return this.kind === "alert_id_stale";
  }

  /** Whether the backend refused to serve, rather than failed to find. */
  get isRefusal(): boolean {
    return this.kind === "provenance_refused" || this.kind === "artifact_missing";
  }
}

async function request<T>(path: string, signal?: AbortSignal): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      // Without this the session cookie is not sent and every analytical
      // call would 401 while looking like a permissions bug.
      credentials: "same-origin",
      signal,
      headers: { Accept: "application/json" },
    });
  } catch (cause) {
    if ((cause as Error)?.name === "AbortError") throw cause;
    throw new ApiError(
      "network",
      0,
      "Could not reach the ObsidianChain API. Start it with " +
        "`make serve` and reload.",
    );
  }

  if (!response.ok) {
    let kind: ApiErrorKind = "unknown";
    let detail = `Request failed with status ${response.status}.`;
    try {
      const body = await response.json();
      if (typeof body?.error === "string") kind = body.error as ApiErrorKind;
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // A non-JSON body is itself informative: the backend always sends
      // JSON, so this is a proxy or a crash rather than a handled refusal.
    }
    if (kind === "unknown" && response.status === 503) kind = "artifact_missing";
    reportUnauthenticated(response.status);
    throw new ApiError(kind, response.status, detail);
  }
  return (await response.json()) as T;
}

export function buildAlertQuery(filters: AlertFilters): string {
  const params = new URLSearchParams();
  for (const severity of filters.severity ?? []) params.append("severity", severity);
  if (filters.minRisk !== undefined) params.set("min_risk", String(filters.minRisk));
  if (filters.maxRisk !== undefined) params.set("max_risk", String(filters.maxRisk));
  if (filters.firstTimestep !== undefined) {
    params.set("first_timestep", String(filters.firstTimestep));
  }
  if (filters.lastTimestep !== undefined) {
    params.set("last_timestep", String(filters.lastTimestep));
  }
  params.set("limit", String(filters.limit ?? 25));
  params.set("offset", String(filters.offset ?? 0));
  return params.toString();
}

export function fetchAlerts(
  filters: AlertFilters,
  signal?: AbortSignal,
): Promise<AlertListResponse> {
  return request<AlertListResponse>(`/alerts?${buildAlertQuery(filters)}`, signal);
}

export function fetchAlert(
  alertId: string,
  signal?: AbortSignal,
): Promise<AlertDetail> {
  return request<AlertDetail>(`/alerts/${encodeURIComponent(alertId)}`, signal);
}


/**
 * Upload one bulk metadata file for validation.
 *
 * The body is the file's raw bytes, not multipart: `python-multipart` is not
 * in the vendored wheel set and the backend image builds with no network, so
 * a raw body keeps the offline guarantee and needs no new dependency.
 *
 * This validates and correlates. It does NOT score - the response says so in
 * its own body, and the UI repeats it rather than implying an upload
 * produced the alerts on screen.
 */
export async function ingestFile(file: File, format?: string): Promise<IngestResult> {
  const params = new URLSearchParams({ filename: file.name });
  if (format) params.set("format", format);
  let response: Response;
  try {
    response = await fetch(`${BASE}/ingest?${params}`, {
      method: "POST",
      credentials: "same-origin",
      body: file,
      headers: { "Content-Type": "application/octet-stream" },
    });
  } catch {
    throw new ApiError("network", 0,
      "Could not reach the ObsidianChain API. Start it with `make serve`.");
  }
  if (!response.ok) {
    let kind: ApiErrorKind = "unknown";
    let detail = `Upload failed with status ${response.status}.`;
    try {
      const body = await response.json();
      if (typeof body?.error === "string") kind = body.error as ApiErrorKind;
      if (typeof body?.detail === "string") detail = body.detail;
    } catch { /* non-JSON body: a proxy or a crash, not a handled refusal */ }
    reportUnauthenticated(response.status);
    throw new ApiError(kind, response.status, detail);
  }
  return (await response.json()) as IngestResult;
}
