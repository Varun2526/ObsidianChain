import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import * as api from "../../api/console";
import type { Investigation, InvestigationStatus } from "../../api/types";
import { useAuth } from "../../store/auth";
import { RunStatusChip } from "../../components/layout/CaseChrome";
import { ErrorState } from "../../components/ui/ErrorState";

/**
 * Cases come from `/api/investigations`, which returns only what this user
 * may see. There is no client-side filtering of someone else's cases,
 * because they never arrive in the first place.
 *
 * There is also no delete. A case is CLOSED, never removed - deleting one
 * would destroy the audit trail that makes its output defensible.
 */

/* ── Status badge colour mapping ─────────────────────────────────── */

/* ── All 10 backend lifecycle states in chronological order ──────── */

const LIFECYCLE_STATUSES: { value: InvestigationStatus; label: string }[] = [
  { value: "DRAFT", label: "Draft" },
  { value: "VALIDATING", label: "Validating" },
  { value: "ANALYZING", label: "Analyzing" },
  { value: "ACTIVE", label: "Active" },
  { value: "SUBMITTED", label: "Submitted" },
  { value: "IN_REVIEW", label: "In Review" },
  { value: "RETURNED", label: "Returned" },
  { value: "APPROVED", label: "Approved" },
  { value: "CLOSED", label: "Closed" },
  { value: "ARCHIVED", label: "Archived" },
];

/* ── Helpers ─────────────────────────────────────────────────────── */

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 30) return `${days}d ago`;
  return new Date(iso).toLocaleDateString();
}

/* ── Component ───────────────────────────────────────────────────── */

export function InvestigationsPage() {
  const { can } = useAuth();
  const [investigations, setInvestigations] = useState<Investigation[]>([]);
  const [scope, setScope] = useState<"all" | "owned">("owned");
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<InvestigationStatus | "ALL">("ALL");
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    api.listInvestigations(controller.signal)
      .then((r) => { setInvestigations(r.investigations); setScope(r.scope); })
      .catch((cause) => {
        if ((cause as Error)?.name !== "AbortError") setError(cause);
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, []);

  /* ── Derived data ────────────────────────────────────────────── */

  const statusCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const inv of investigations) {
      counts[inv.status] = (counts[inv.status] ?? 0) + 1;
    }
    return counts;
  }, [investigations]);

  const activeCount = useMemo(() => {
    return (statusCounts["ACTIVE"] ?? 0) + (statusCounts["VALIDATING"] ?? 0) + (statusCounts["ANALYZING"] ?? 0);
  }, [statusCounts]);

  const reviewCount = useMemo(() => {
    return (statusCounts["IN_REVIEW"] ?? 0) + (statusCounts["SUBMITTED"] ?? 0) + (statusCounts["RETURNED"] ?? 0);
  }, [statusCounts]);

  const totalOutstanding = useMemo(
    () => investigations.reduce((n, i) => n + (i.summary?.outstanding ?? 0), 0),
    [investigations],
  );

  const q = search.trim().toLowerCase();
  const filtered = useMemo(() => {
    let list = investigations;
    if (statusFilter !== "ALL") {
      list = list.filter((i) => i.status === statusFilter);
    }
    if (q) {
      list = list.filter(
        (i) =>
          i.name.toLowerCase().includes(q) ||
          i.case_label.toLowerCase().includes(q) ||
          (i.owner?.display_name ?? "").toLowerCase().includes(q),
      );
    }
    return list;
  }, [investigations, statusFilter, q]);

  /* ── Render ──────────────────────────────────────────────────── */

  return (
    <>
      {/* Page header */}
      <div className="page-header">
        <div>
          <h1>Investigations</h1>
          <p className="muted" style={{ margin: 0 }}>
            {scope === "all"
              ? "All cases on this workstation — your role grants oversight"
              : "Cases you own"}
          </p>
        </div>
        {can("create_investigation") && (
          <Link to="/investigations/new" className="btn btn-primary">
            + New Investigation
          </Link>
        )}
      </div>

      {/* Summary stats */}
      {!loading && !error && investigations.length > 0 && (
        <div className="inv-stats-row">
          <div className="inv-stat">
            <span className="inv-stat-value">{investigations.length}</span>
            <span className="inv-stat-label">Total Cases</span>
          </div>
          <div className="inv-stat">
            <span className="inv-stat-value" style={{ color: "var(--model)" }}>
              {activeCount}
            </span>
            <span className="inv-stat-label">Active</span>
          </div>
          <div className="inv-stat">
            <span className="inv-stat-value" style={{ color: "var(--blockchain)" }}>
              {reviewCount}
            </span>
            <span className="inv-stat-label">In Review</span>
          </div>
          <div className="inv-stat">
            <span className="inv-stat-value" style={{ color: "var(--high)" }}>
              {totalOutstanding}
            </span>
            <span className="inv-stat-label">Outstanding Alerts</span>
          </div>
        </div>
      )}

      {/* Search + filter bar */}
      <div className="inv-toolbar">
        <div className="inv-search">
          <span className="inv-search-icon">⌕</span>
          <input
            type="text"
            placeholder="Search by case name, number, or owner…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <div className="inv-filters">
          <label htmlFor="inv-status-filter" className="inv-filter-label">
            Status:
          </label>
          <select
            id="inv-status-filter"
            className="inv-status-select"
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as InvestigationStatus | "ALL")}
          >
            <option value="ALL">All Statuses ({investigations.length})</option>
            {LIFECYCLE_STATUSES.map(({ value, label }) => {
              const count = statusCounts[value] ?? 0;
              return (
                <option key={value} value={value}>
                  {label} ({count})
                </option>
              );
            })}
          </select>
          {statusFilter !== "ALL" && (
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => setStatusFilter("ALL")}
              title="Reset status filter"
            >
              Reset
            </button>
          )}
        </div>
      </div>

      {/* Table */}
      <section className="panel">
        {error ? (
          <ErrorState error={error} />
        ) : loading ? (
          <div className="skeleton">
            <div className="skeleton-row" />
            <div className="skeleton-row" />
            <div className="skeleton-row" />
            <div className="skeleton-row" />
          </div>
        ) : filtered.length === 0 ? (
          <div className="empty">
            <h3>{investigations.length === 0 ? "No investigations yet" : "No matches"}</h3>
            <p>
              {investigations.length === 0
                ? "Create a new investigation to begin your analysis."
                : "Try adjusting your search or filters."}
            </p>
          </div>
        ) : (
          <div className="panel-body flush">
            <table>
              <thead>
                <tr>
                  <th>Case</th>
                  <th>Name</th>
                  <th>Owner</th>
                  <th>Status</th>
                  <th className="num">Alerts</th>
                  <th className="num">Outstanding</th>
                  <th>Analytical Run</th>
                  <th>Updated</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {filtered.map((inv) => (
                  <tr key={inv.id}>
                    <td>
                      <span className="inv-case-label">{inv.case_label}</span>
                    </td>
                    <td>
                      <Link to={`/inv/${inv.id}`} className="inv-name-link">
                        {inv.name}
                      </Link>
                    </td>
                    <td className="small muted">
                      {inv.owner?.display_name ?? inv.owner?.username ?? "—"}
                    </td>
                    <td>
                      <span className={`status-badge status-${inv.status.toLowerCase()}`}>
                        {inv.status.replace("_", " ")}
                      </span>
                    </td>
                    <td className="num">{inv.summary?.alerts_referenced ?? 0}</td>
                    <td className="num">
                      {(inv.summary?.outstanding ?? 0) > 0 ? (
                        <span style={{ color: "var(--high)", fontWeight: 600 }}>
                          {inv.summary?.outstanding}
                        </span>
                      ) : (
                        <span className="muted">0</span>
                      )}
                    </td>
                    <td><RunStatusChip status={inv.run_status ?? "UNBOUND"} /></td>
                    <td className="small muted" title={new Date(inv.updated_at).toLocaleString()}>
                      {relativeTime(inv.updated_at)}
                    </td>
                    <td>
                      <Link to={`/inv/${inv.id}`} className="btn btn-sm">
                        Open →
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="table-footer">
              <span className="small muted">
                Showing {filtered.length} of {investigations.length} investigation{investigations.length !== 1 ? "s" : ""}
              </span>
            </div>
          </div>
        )}
      </section>
    </>
  );
}
