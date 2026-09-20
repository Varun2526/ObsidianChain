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
      <div className="login-backdrop" />
      <form className="login-box institutional-login" onSubmit={submit}>
        <div className="login-brand">
          <div className="institutional-monogram">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
              <path d="M12 2L3 7v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-9-5z" />
              <path d="M12 7v10" />
              <path d="M8.5 10.5l7 3" />
              <path d="M15.5 10.5l-7 3" />
            </svg>
          </div>
          <h1 className="institutional-title">OBSIDIANCHAIN</h1>
          <p className="institutional-subtitle">BITCOIN TRANSACTION TRAFFIC INVESTIGATION PLATFORM</p>
          <div className="institutional-pills">
            <span>Secure</span>
            <span className="dot">•</span>
            <span>Offline</span>
            <span className="dot">•</span>
            <span>Evidence-Driven</span>
          </div>
        </div>

        <div className="login-fields">
          <div className="field-group">
            <label htmlFor="inv-id">Username</label>
            <input
              id="inv-id"
              type="text"
              autoFocus
              autoComplete="username"
              value={username}
              onChange={(e) => { setUsername(e.target.value); setError(null); }}
              placeholder="e.g. investigator, reviewer, or admin"
            />
          </div>

          <div className="field-group">
            <label htmlFor="inv-pw">Password</label>
            <input
              id="inv-pw"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => { setPassword(e.target.value); setError(null); }}
              placeholder="••••••••••••"
            />
          </div>


          {error && (
            <div className="login-error-banner">
              <span className="error-icon">⚠</span>
              <span>{error}</span>
            </div>
          )}

          <button type="submit" className="login-btn primary-action" disabled={busy || !username || !password}>
            {busy ? "Authenticating Session…" : "Sign in securely"}
          </button>

          <div className="login-security-footer">
            <div className="security-badge">
              <span className="secure-dot" />
              <span>OFFLINE AIR-GAPPED ENVIRONMENT</span>
            </div>
            <p className="security-caption">
              ObsidianChain forensic node · Immutable audit logging enabled
            </p>
          </div>
        </div>
      </form>
    </div>
  );
}
