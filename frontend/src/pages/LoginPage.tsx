import { useEffect, useState, type FormEvent } from "react";
import { useAuth } from "../store/auth";
import { ApiError } from "../api/client";
import * as console_api from "../api/console";
import type { Role } from "../api/types";

/**
 * The password is sent to the backend and verified there.
 *
 * The previous version of this form accepted any non-empty investigator id
 * and discarded the password. Now a failure is the backend's 401, rendered
 * in the backend's own words - deliberately identical for an unknown
 * username, a wrong password and a deactivated account, so this form cannot
 * be used to find out which accounts exist.
 */
export function LoginPage() {
  const { login, demoLogin } = useAuth();
  // Offered only when the server runs with OBSIDIANCHAIN_DEMO_LOGIN=1.
  const [demoRoles, setDemoRoles] = useState<console_api.DemoRole[]>([]);
  useEffect(() => {
    const ctrl = new AbortController();
    console_api.demoStatus(ctrl.signal)
      .then((r) => { if (r.enabled) setDemoRoles(r.roles); })
      .catch(() => {});
    return () => ctrl.abort();
  }, []);

  const enterAs = async (role: Role) => {
    setBusy(true);
    setError(null);
    try {
      await demoLogin(role);
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.detail : "Could not reach the ObsidianChain API.");
    } finally {
      setBusy(false);
    }
  };
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(username, password);
    } catch (cause) {
      setError(
        cause instanceof ApiError
          ? cause.detail
          : "Could not reach the ObsidianChain API. Start it with `make serve`.",
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login-page">
      <aside className="login-aside" aria-hidden="true">
        <span className="wordmark" style={{ fontSize: 16 }}>Obsidian<b>Chain</b></span>
        <div>
          <h1>Trace where value came from and where it went, and keep what is known apart from what is inferred.</h1>
          <ul className="login-principles">
            <li><span className="ev ev-chain">On-chain</span><span>Observed transactions and flows, true independent of any model.</span></li>
            <li><span className="ev ev-watchlist">Attribution</span><span>External watchlists, named and never restated as findings.</span></li>
            <li><span className="ev ev-model">Model</span><span>Learned risk associations with their evaluation record. A lead, not proof.</span></li>
          </ul>
        </div>
        <span className="small faint">Runs offline. Every decision is written to an append-only audit log.</span>
      </aside>

      <main className="login-main">
        <form className="login-box" onSubmit={submit}>
          <div className="login-brand">
            <h1>Sign in</h1>
            <p>Blockchain investigation and risk intelligence</p>
          </div>

          {demoRoles.length > 0 && (
            <section className="demo-access" aria-label="Demo access">
              <div className="demo-access-head">
                <strong>Demo access</strong>
                <span className="small faint">Choose a role to enter without a password</span>
              </div>
              <div className="demo-access-roles">
                {demoRoles.map((r) => (
                  <button key={r.role} type="button" className="demo-role" disabled={busy}
                          onClick={() => enterAs(r.role)}>
                    <span className="demo-role-name">{r.display_name}</span>
                    <span className="demo-role-tag">{r.role.toLowerCase()}</span>
                    <span className="demo-role-desc">{r.description}</span>
                  </button>
                ))}
              </div>
              <p className="small faint" style={{ margin: 0 }}>
                Every action is still audited under the chosen role. Or sign in with an account below.
              </p>
            </section>
          )}

          <div className="login-fields">
            <label htmlFor="inv-id">Username
              <input
                id="inv-id"
                type="text"
                autoFocus={demoRoles.length === 0}
                autoComplete="username"
                value={username}
                onChange={(e) => { setUsername(e.target.value); setError(null); }}
              />
            </label>

            <label htmlFor="inv-pw">Password
              <input
                id="inv-pw"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => { setPassword(e.target.value); setError(null); }}
              />
            </label>

            {error && (
              <div className="login-error-banner" role="alert">{error}</div>
            )}

            <button type="submit" className="btn btn-primary login-btn" disabled={busy || !username || !password}>
              {busy ? "Signing in…" : "Sign in"}
            </button>
          </div>

          <p className="login-security-footer">
            <span className="security-badge"><span className="secure-dot" />Offline deployment</span>
            <span className="security-caption">Accounts are issued by an administrator. Failed attempts are rate limited and audited.</span>
          </p>
        </form>
      </main>
    </div>
  );
}
