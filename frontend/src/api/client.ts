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
 */

import type {
  AlertDetail,
  AlertFilters,
  AlertListResponse,
  ApiErrorKind,
} from "./types";

const BASE = "/api";

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
