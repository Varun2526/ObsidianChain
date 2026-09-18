import { useEffect, useState } from "react";
import { useAuth } from "../store/auth";

/**
 * Identity, role and the authorisation policy, shown as the backend reports
 * them.
 *
 * The capability list is read from `/api/auth/me`, which re-derives it from
 * the session's role on every request. It is displayed so a person can see
 * what they may do - it is not what decides it. Editing this page's state
 * changes what is drawn and nothing else.
 */
export function SettingsPage() {
  const { identity } = useAuth();
  const [policy, setPolicy] = useState<Record<string, string[]> | null>(null);

  useEffect(() => {
    fetch("/api/roles", { credentials: "same-origin" })
      .then((r) => r.json())
      .then((b) => setPolicy(b.roles))
      .catch(() => {});
  }, []);

  return (
    <>
      <div className="page-header"><h1>Settings</h1></div>

      <section className="panel">
        <div className="panel-head"><h2>Signed in as</h2></div>
        <div className="panel-body">
          <dl className="kv">
            <dt>Username</dt><dd className="mono">{identity?.user.username}</dd>
            <dt>Display name</dt><dd>{identity?.user.display_name}</dd>
            <dt>Role</dt><dd>{identity?.user.role}</dd>
            <dt>Account created</dt>
            <dd>
              {identity?.user.created_at
                ? new Date(identity.user.created_at).toLocaleString() : "—"}
            </dd>
          </dl>
          <p className="note">
            The session is a server-side record; the browser holds only an
            HttpOnly cookie it cannot read. Signing out revokes the session in
            the database, so a copied cookie authenticates nothing afterwards.
          </p>
        </div>
      </section>

      <section className="panel">
        <div className="panel-head"><h2>Your capabilities</h2></div>
        <div className="panel-body">
          <div className="chips">
            {(identity?.capabilities ?? []).map((c) => (
              <span key={c} className="cat cat-BLOCKCHAIN_CONTEXT">{c}</span>
            ))}
          </div>
          <p className="note">
            Shown for orientation. Every protected operation is re-checked
            against the session's role on the server; this list is not the
            security boundary. Case-scoped actions additionally require
            ownership.
          </p>
        </div>
      </section>

      {policy && (
        <section className="panel">
          <div className="panel-head"><h2>Authorisation policy</h2></div>
          <div className="panel-body flush">
            <table>
              <thead><tr><th>Role</th><th>Capabilities</th></tr></thead>
              <tbody>
                {Object.entries(policy).map(([role, caps]) => (
                  <tr key={role}>
                    <td className="mono">{role}</td>
                    <td className="small muted">{caps.join(", ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section className="panel">
        <div className="panel-head"><h2>System</h2></div>
        <div className="panel-body">
          <dl className="kv">
            <dt>Application</dt><dd>ObsidianChain Investigation Console</dd>
            <dt>API endpoint</dt><dd className="mono">/api</dd>
            <dt>Stack</dt><dd>React 18 · TypeScript · Vite · FastAPI · SQLite</dd>
          </dl>
          <p className="note" style={{ marginTop: 12 }}>
            Analytical results — clustering, risk scores, explanations and
            evidence — are produced by the offline pipeline and read from
            immutable artifacts. This console records investigator state
            beside them and never writes to them. No external resource is
            fetched at any point; the application runs with the network
            disabled.
          </p>
        </div>
      </section>
    </>
  );
}
