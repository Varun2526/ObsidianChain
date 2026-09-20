import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import * as api from "../../api/console";
import type { Investigation } from "../../api/types";
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
export function InvestigationsPage() {
  const { can } = useAuth();
  const [investigations, setInvestigations] = useState<Investigation[]>([]);
  const [scope, setScope] = useState<"all" | "owned">("owned");
  const [search, setSearch] = useState("");
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

  const q = search.trim().toLowerCase();
  const filtered = q
    ? investigations.filter(
        (i) =>
          i.name.toLowerCase().includes(q) ||
          i.case_label.toLowerCase().includes(q),
      )
    : investigations;

  return (
    <>
      <div className="page-header">
        <div>
          <h1>Investigations</h1>
          <p className="muted">
            {scope === "all"
              ? "Every case on this workstation (your role grants oversight)"
              : "Cases you own"}
          </p>
        </div>
        {can("create_investigation") && (
          <Link to="/investigations/new" className="btn btn-primary">
            + New Investigation
          </Link>
        )}
      </div>

      <div className="search-bar">
        <input
          type="text"
          placeholder="Search by case name or number…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      <section className="panel">
        {error ? (
          <ErrorState error={error} />
        ) : loading ? (
          <div className="panel-body"><p className="muted">Loading…</p></div>
        ) : filtered.length === 0 ? (
          <div className="panel-body">
            <p className="muted">
              {investigations.length === 0
                ? "No investigations yet."
                : "No investigations match your search."}
            </p>
          </div>
        ) : (
          <div className="panel-body flush">
            <table>
              <thead>
                <tr>
                  <th>Case</th><th>Name</th><th>Owner</th><th>Status</th>
                  <th className="num">Alerts</th><th className="num">Outstanding</th>
                  <th>Analytical run</th><th>Updated</th><th />
                </tr>
              </thead>
              <tbody>
                {filtered.map((inv) => (
                  <tr key={inv.id}>
                    <td className="mono">{inv.case_label}</td>
                    <td>{inv.name}</td>
                    <td className="small muted">
                      {inv.owner?.display_name ?? inv.owner?.username ?? "—"}
                    </td>
                    <td>
                      <span className={`status-badge status-${inv.status.toLowerCase()}`}>
                        {inv.status}
                      </span>
                    </td>
                    <td className="num">{inv.summary?.alerts_referenced ?? 0}</td>
                    <td className="num">{inv.summary?.outstanding ?? 0}</td>
                    <td><RunStatusChip status={inv.run_status ?? "UNBOUND"} /></td>
                    <td className="small muted">
                      {new Date(inv.updated_at).toLocaleDateString()}
                    </td>
                    <td>
                      <Link to={`/inv/${inv.id}`} className="btn btn-sm">Open</Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}
