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
      <form className="login-box" onSubmit={submit}>
        <div className="login-brand">
          <span className="login-logo" />
          <h1>OBSIDIANCHAIN</h1>
          <p>Bitcoin Transaction Intelligence</p>
        </div>

        <div className="login-fields">
          <label htmlFor="inv-id">Username</label>
          <input
            id="inv-id"
            type="text"
            autoFocus
            autoComplete="username"
            value={username}
            onChange={(e) => { setUsername(e.target.value); setError(null); }}
            placeholder="Enter your username"
          />

          <label htmlFor="inv-pw">Password</label>
          <input
            id="inv-pw"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => { setPassword(e.target.value); setError(null); }}
            placeholder="Enter password"
          />

          {error && <p className="login-error">{error}</p>}

          <button type="submit" className="login-btn" disabled={busy}>
            {busy ? "Signing in…" : "Sign In"}
          </button>

          <p className="login-note">
            Accounts are created by an administrator on this workstation:{" "}
            <code>obsidianchain console-user-add &lt;username&gt; --role ADMIN</code>
          </p>
        </div>
      </form>
    </div>
  );
}
