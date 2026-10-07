/**
 * The application-layer client: identity, cases, casework, reports, audit.
 *
 * Two things this file is careful about.
 *
 * First, `credentials: "same-origin"` on every call. The session is an
 * HttpOnly cookie the browser holds and this code cannot read, which is the
 * point: there is no token in localStorage to steal, copy or edit, and the
 * backend re-reads the session row on every request.
 *
 * Second, the error taxonomy is preserved rather than flattened. The console
 * distinguishes 401 (not authenticated) from 403 (authenticated, not
 * allowed) from 404 (no such case) from 409 (the analytical run moved under
 * you), and each of those means something different to the person at the
 * keyboard. Collapsing them to "something went wrong" would hide exactly the
 * facts an investigator needs.
 */

import { ApiError, reportUnauthenticated } from "./client";
import type {
  AuditEvent,
  CaseAlertDetail,
  CaseAlertRow,
  Disposition,
  Identity,
  Investigation,
  InvestigatorNote,
  ReportPayload,
  AlertPatterns,
  RelatedAlertsResponse,
  SavedFilter,
  SeparationEvidence,
  TransactionDrilldown,
  UploadedDataset,
  RunProgressResponse,
  Role,
  UserAccount,
} from "./types";

export type { SavedFilter, RelatedAlertsResponse };

const BASE = "/api";

async function call<T>(
  path: string,
  init: RequestInit = {},
  signal?: AbortSignal,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      // Without this the cookie is not sent and every authenticated call
      // would 401 while looking like a permissions bug.
      credentials: "same-origin",
      signal,
      ...init,
      headers: { Accept: "application/json", ...(init.headers ?? {}) },
    });
  } catch (cause) {
    if ((cause as Error)?.name === "AbortError") throw cause;
    throw new ApiError(
      "network",
      0,
      "Could not reach the ObsidianChain API. Start it with `make serve` and reload.",
    );
  }

  if (!response.ok) {
    let kind = "unknown";
    let detail = `Request failed with status ${response.status}.`;
    try {
      const body = await response.json();
      if (typeof body?.error === "string") kind = body.error;
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON body: a proxy or a crash, not a handled refusal */
    }
    // One channel for the whole application; see client.ts. Only 401 - a
    // 403 means the session is fine and this user may not have that thing.
    reportUnauthenticated(response.status);
    throw new ApiError(kind as never, response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

function post<T>(path: string, body?: unknown): Promise<T> {
  return call<T>(path, {
    method: "POST",
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

/** True when the backend refused because nobody is logged in. */
export function isUnauthenticated(error: unknown): boolean {
  return (
    error instanceof ApiError &&
    error.status === 401
  );
}

/** True when the backend refused an authenticated caller. */
export function isForbidden(error: unknown): boolean {
  return error instanceof ApiError && error.status === 403;
}

export function isNotFound(error: unknown): boolean {
  return error instanceof ApiError && error.status === 404;
}

// ---- identity -----------------------------------------------------------

export const login = (username: string, password: string) =>
  post<Identity>("/auth/login", { username, password });

export const logout = () => post<{ ok: boolean }>("/auth/logout");

/** Liveness, and whether the Elliptic++ reference artifacts are installed. */
export const health = (signal?: AbortSignal) =>
  call<{ status: string; version: string; artifacts_present: boolean }>("/health", {}, signal);

/** One-click demo sign-in, offered only when the server enables it. */
export interface DemoRole { role: Role; username: string; display_name: string; description: string }
export const demoStatus = (signal?: AbortSignal) =>
  call<{ enabled: boolean; roles: DemoRole[] }>("/auth/demo", {}, signal);
export const demoLogin = (role: Role) => post<Identity>("/auth/demo-login", { role });

export const me = (signal?: AbortSignal) =>
  call<Identity>("/auth/me", {}, signal);

export const assignableUsers = () =>
  call<{ users: { id: string; username: string; display_name: string; role: string }[] }>(
    "/users/assignable",
  );

export const listUsers = (signal?: AbortSignal) =>
  call<{ users: UserAccount[] }>("/users", {}, signal);

export const createUser = (payload: {
  username: string;
  password: string;
  display_name: string;
  role: string;
}) =>
  post<UserAccount>("/users", payload);

export const deactivateUser = (userId: string) =>
  post<{ ok: boolean }>(`/users/${encodeURIComponent(userId)}/deactivate`);

export const activateUser = (userId: string) =>
  post<{ ok: boolean }>(`/users/${encodeURIComponent(userId)}/activate`);

export const setUserRole = (userId: string, role: string) =>
  post<{ ok: boolean }>(`/users/${encodeURIComponent(userId)}/role`, { role });

export const resetUserPassword = (userId: string, newPassword: string) =>
  post<{ ok: boolean }>(`/users/${encodeURIComponent(userId)}/reset-password`, {
    new_password: newPassword,
  });

export const deleteUser = (userId: string) =>
  call<{ ok: boolean }>(`/users/${encodeURIComponent(userId)}`, {
    method: "DELETE",
  });


// ---- investigations -----------------------------------------------------

export const listInvestigations = (signal?: AbortSignal) =>
  call<{
    investigations: Investigation[];
    scope: "all" | "owned";
    current_artifact_run: string | null;
  }>("/investigations", {}, signal);

export const createInvestigation = (name: string, description: string) =>
  post<Investigation>("/investigations", { name, description });

export const getInvestigation = (id: string, signal?: AbortSignal) =>
  call<Investigation>(`/investigations/${encodeURIComponent(id)}`, {}, signal);

export const patchInvestigation = (
  id: string,
  patch: { name?: string; description?: string },
) =>
  call<Investigation>(`/investigations/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });

export const setInvestigationStatus = (id: string, status: string) =>
  post<Investigation>(`/investigations/${encodeURIComponent(id)}/status`, {
    status,
  });

export const archiveInvestigation = (id: string) =>
  post<Investigation>(`/investigations/${encodeURIComponent(id)}/archive`);

export const restoreInvestigation = (id: string) =>
  post<Investigation>(`/investigations/${encodeURIComponent(id)}/restore`);

export const deleteInvestigation = (id: string, confirmation: string) =>
  call<{ ok: boolean }>(
    `/investigations/${encodeURIComponent(id)}?confirmation=${encodeURIComponent(confirmation)}`,
    { method: "DELETE" },
  );

// ---- datasets -----------------------------------------------------------

/**
 * Upload one capture INTO a case.
 *
 * Raw body rather than multipart, for the reason the backend already
 * documents: `python-multipart` is not in the vendored wheel set and the
 * image builds with no network.
 *
 * The response carries an `analysis_run` whose status is NOT_RUN. That is
 * not a placeholder - it is the honest statement that validating a file is
 * not scoring it.
 */
export async function uploadDataset(
  investigationId: string,
  file: File,
  format?: string,
): Promise<{ dataset: UploadedDataset; analysis_run: Record<string, unknown> }> {
  const params = new URLSearchParams({ filename: file.name });
  if (format) params.set("format", format);
  return call(
    `/investigations/${encodeURIComponent(investigationId)}/datasets?${params}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    },
  );
}

export const listDatasets = (investigationId: string) =>
  call<{ datasets: UploadedDataset[] }>(
    `/investigations/${encodeURIComponent(investigationId)}/datasets`,
  );

export const runDatasetAnalysis = (
  investigationId: string,
  datasetId: string,
  options?: Record<string, unknown>,
) =>
  post<{
    run_id: string;
    analysis_run: Record<string, unknown>;
    alerts_count: number;
    manifest: Record<string, unknown>;
  }>(
    `/investigations/${encodeURIComponent(investigationId)}/datasets/${encodeURIComponent(datasetId)}/run`,
    options,
  );

export const getRunProgress = (
  investigationId: string,
  runId: string,
  signal?: AbortSignal,
) =>
  call<RunProgressResponse>(
    `/investigations/${encodeURIComponent(investigationId)}/runs/${encodeURIComponent(runId)}/progress`,
    {},
    signal,
  );


// ---- casework -----------------------------------------------------------

export const listCaseAlerts = (investigationId: string, signal?: AbortSignal) =>
  call<{
    alerts: CaseAlertRow[];
    current_artifact_run: string | null;
    stale_count: number;
    summary: Investigation["summary"];
    meaning: string;
  }>(`/investigations/${encodeURIComponent(investigationId)}/alerts`, {}, signal);

export const referenceAlert = (investigationId: string, alertId: string) =>
  post<{ alert_id: string; run_fingerprint: string; disposition: Disposition | null }>(
    `/investigations/${encodeURIComponent(investigationId)}/alerts`,
    { alert_id: alertId },
  );

export const getCaseAlert = (
  investigationId: string,
  alertId: string,
  signal?: AbortSignal,
) =>
  call<CaseAlertDetail>(
    `/investigations/${encodeURIComponent(investigationId)}/alerts/${encodeURIComponent(alertId)}`,
    {},
    signal,
  );

export const getRunAlert = (investigationId: string, alertRef: string, signal?: AbortSignal) =>
  call<import("./types").RunAlertAnalytical & { referenced: boolean }>(
    `/investigations/${encodeURIComponent(investigationId)}/run-alerts/${encodeURIComponent(alertRef)}`,
    {},
    signal,
  );

export const setDisposition = (
  investigationId: string,
  alertId: string,
  state: string,
  rationale: string,
) =>
  post<{ disposition: Disposition; history: Disposition[] }>(
    `/investigations/${encodeURIComponent(investigationId)}/alerts/${encodeURIComponent(alertId)}/disposition`,
    { state, rationale },
  );

export const assignAlert = (
  investigationId: string,
  alertId: string,
  assignedTo: string | null,
) =>
  post<{ alert_id: string; assigned_to: string | null; assigned_at: string | null }>(
    `/investigations/${encodeURIComponent(investigationId)}/alerts/${encodeURIComponent(alertId)}/assignment`,
    { assigned_to: assignedTo },
  );

// ---- notes --------------------------------------------------------------

export const listNotes = (investigationId: string, alertId?: string) => {
  const query = alertId ? `?alert_id=${encodeURIComponent(alertId)}` : "";
  return call<{ notes: InvestigatorNote[] }>(
    `/investigations/${encodeURIComponent(investigationId)}/notes${query}`,
  );
};

export const addNote = (
  investigationId: string,
  body: string,
  alertId?: string,
) =>
  post<InvestigatorNote>(
    `/investigations/${encodeURIComponent(investigationId)}/notes`,
    alertId ? { body, alert_id: alertId } : { body },
  );

// ---- report -------------------------------------------------------------

export const getReport = (investigationId: string, version?: number) => {
  const query = version ? `?version=${version}` : "";
  return call<ReportPayload>(
    `/investigations/${encodeURIComponent(investigationId)}/report${query}`,
  );
};

export const saveReport = (
  investigationId: string,
  payload: { title: string; executive_summary: string; content: string },
) =>
  post<ReportPayload>(
    `/investigations/${encodeURIComponent(investigationId)}/report`,
    payload,
  );

export const finaliseReport = (investigationId: string, version: number) =>
  post<Record<string, unknown>>(
    `/investigations/${encodeURIComponent(investigationId)}/report/${version}/finalise`,
  );

export const recordExport = (investigationId: string, version: number) =>
  post<{ ok: boolean; content_sha256: string }>(
    `/investigations/${encodeURIComponent(investigationId)}/report/${version}/export`,
  );

// ---- audit --------------------------------------------------------------

export const getHistory = (investigationId: string, signal?: AbortSignal) =>
  call<{ events: AuditEvent[]; append_only: boolean }>(
    `/investigations/${encodeURIComponent(investigationId)}/history`,
    {},
    signal,
  );

// ---- separation evidence (analytical, read-only) ------------------------

/**
 * The network-separation records for one alert's cluster.
 *
 * Analytical, not case state - it lives on the read-only side and needs no
 * session. Surfaced here because nothing called it before: the separation
 * layer is the project's actual contribution and was invisible in the UI.
 */
export const getSeparationEvidence = (alertId: string, signal?: AbortSignal) =>
  call<SeparationEvidence>(
    `/alerts/${encodeURIComponent(alertId)}/separation-evidence`,
    {},
    signal,
  );

/**
 * Peeling-chain and mixing-like structure behind one alert.
 *
 * Analytical and read-only, like the separation evidence. Both structures
 * are OBSERVATIONS about transaction shape: neither establishes laundering,
 * ownership or an offence, and the response carries the wording that says so.
 */
export const getAlertPatterns = (alertId: string, signal?: AbortSignal) =>
  call<AlertPatterns>(
    `/alerts/${encodeURIComponent(alertId)}/patterns`,
    {},
    signal,
  );

/**
 * The controlled synthetic evaluation.
 *
 * SYNTHETIC_CONTROL throughout, served from a path no case reads. Nothing it
 * returns describes the production analytical run.
 */
export const getSyntheticEvaluation = (signal?: AbortSignal) =>
  call<Record<string, unknown>>("/evaluation/synthetic", {}, signal);

// ---- recent activity (T1.1) ---------------------------------------------

export const recentActivity = (limit = 25, signal?: AbortSignal) =>
  call<{ events: AuditEvent[] }>(
    `/investigations/activity/recent?limit=${limit}`,
    {},
    signal,
  );

// ---- saved filters (T2.7) -----------------------------------------------

export const listSavedFilters = (investigationId: string, signal?: AbortSignal) =>
  call<{ filters: SavedFilter[] }>(
    `/investigations/${encodeURIComponent(investigationId)}/filters`,
    {},
    signal,
  );

export const saveFilter = (
  investigationId: string,
  name: string,
  filter: unknown,
) =>
  post<{ filter: SavedFilter }>(
    `/investigations/${encodeURIComponent(investigationId)}/filters`,
    { name, filter },
  );

export const deleteSavedFilter = (investigationId: string, filterId: string) =>
  call<{ ok: boolean }>(
    `/investigations/${encodeURIComponent(investigationId)}/filters/${encodeURIComponent(filterId)}`,
    { method: "DELETE" },
  );

// ---- export (T2.11) -----------------------------------------------------

export const exportInvestigation = (investigationId: string) =>
  call<Record<string, unknown>>(
    `/investigations/${encodeURIComponent(investigationId)}/export`,
  );

// ---- related alerts (T2.1) ----------------------------------------------

export const getRelatedAlerts = (alertId: string, signal?: AbortSignal) =>
  call<RelatedAlertsResponse>(
    `/alerts/${encodeURIComponent(alertId)}/related`,
    {},
    signal,
  );

// ---- transaction drilldown (T2.4) ---------------------------------------

export const getTransactionDrilldown = (txid: number | string, signal?: AbortSignal) =>
  call<TransactionDrilldown>(
    `/transactions/${encodeURIComponent(String(txid))}`,
    {},
    signal,
  );

export const getRunResults = (investigationId: string, runId: string, limit = 50, signal?: AbortSignal) =>
  call<import("./types").RunResults>(
    `/investigations/${encodeURIComponent(investigationId)}/runs/${encodeURIComponent(runId)}/results?limit=${limit}`,
    {},
    signal,
  );

/** The shared authenticated GET, for the typed modules beside this one (api/intel.ts). */
export function apiGet<T>(path: string, signal?: AbortSignal): Promise<T> {
  return call<T>(path, {}, signal);
}
