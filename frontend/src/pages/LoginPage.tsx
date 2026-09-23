import { useState, type FormEvent } from "react";
import { useAuth } from "../store/auth";
import { ApiError } from "../api/client";

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
  const { login } = useAuth();
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

          <div className="login-fields">
            <label htmlFor="inv-id">Username
              <input
                id="inv-id"
                type="text"
                autoFocus
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
