import { useState } from "react";
import { useAuth } from "../store/auth";

/**
 * Account information and system details for the signed-in user.
 *
 * Internal RBAC capability identifiers and the full authorisation policy
 * matrix are intentionally omitted from this view. They remain enforced
 * server-side and are available to the Admin workspace.
 */

type SettingsTab = "profile" | "security" | "about";

const TABS: { key: SettingsTab; label: string; icon: string }[] = [
  { key: "profile", label: "Profile", icon: "👤" },
  { key: "security", label: "Security", icon: "🔒" },
  { key: "about", label: "About", icon: "ℹ" },
];

const ROLE_LABELS: Record<string, string> = {
  ADMIN: "System Administrator",
  INVESTIGATOR: "Lead Investigator",
  REVIEWER: "Quality Reviewer",
};

export function SettingsPage() {
  const { identity, logout } = useAuth();
  const [activeTab, setActiveTab] = useState<SettingsTab>("profile");

  return (
    <>
      <div className="page-header">
        <div>
          <h1>Settings</h1>
          <p className="muted" style={{ margin: 0 }}>
            Account information, security, and system details
          </p>
        </div>
      </div>

      {/* Tab navigation */}
      <div className="tab-bar" style={{ marginBottom: 24 }}>
        {TABS.map((t) => (
          <button
            key={t.key}
            className={`tab${activeTab === t.key ? " active" : ""}`}
            onClick={() => setActiveTab(t.key)}
          >
            <span style={{ marginRight: 6 }}>{t.icon}</span>
            {t.label}
          </button>
        ))}
      </div>

      {/* ── PROFILE TAB ──────────────────────────────── */}
      {activeTab === "profile" && (
        <>
          <section className="panel">
            <div className="panel-head"><h2>Identity</h2></div>
            <div className="panel-body">
              <div className="settings-profile-grid">
                <div className="settings-avatar">
                  <span className="settings-avatar-circle">
                    {(identity?.user.display_name ?? identity?.user.username ?? "?")
                      .charAt(0)
                      .toUpperCase()}
                  </span>
                </div>
                <div className="settings-profile-info">
                  <dl className="kv">
                    <dt>Username</dt>
                    <dd className="mono">{identity?.user.username}</dd>
                    <dt>Display name</dt>
                    <dd>{identity?.user.display_name}</dd>
                    <dt>Role</dt>
                    <dd>
                      <span className={`status-badge status-${identity?.user.role?.toLowerCase()}`}>
                        {identity?.user.role}
                      </span>
                      <span className="muted small" style={{ marginLeft: 8 }}>
                        {ROLE_LABELS[identity?.user.role ?? ""] ?? identity?.user.role}
                      </span>
                    </dd>
                    <dt>Account created</dt>
                    <dd>
                      {identity?.user.created_at
                        ? new Date(identity.user.created_at).toLocaleString()
                        : "—"}
                    </dd>
                    <dt>Account status</dt>
                    <dd>
                      <span
                        style={{
                          display: "inline-flex",
                          alignItems: "center",
                          gap: 6,
                          color: identity?.user.active ? "var(--model)" : "var(--critical)",
                        }}
                      >
                        <span
                          style={{
                            width: 6,
                            height: 6,
                            borderRadius: "50%",
                            background: identity?.user.active ? "var(--model)" : "var(--critical)",
                            boxShadow: `0 0 6px ${identity?.user.active ? "var(--model)" : "var(--critical)"}`,
                          }}
                        />
                        {identity?.user.active ? "Active" : "Inactive"}
                      </span>
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </section>
        </>
      )}

      {/* ── SECURITY TAB ─────────────────────────────── */}
      {activeTab === "security" && (
        <>
          <section className="panel">
            <div className="panel-head"><h2>Active Session</h2></div>
            <div className="panel-body">
              <dl className="kv">
                <dt>Signed in as</dt>
                <dd className="mono">{identity?.user.username}</dd>
                <dt>Role</dt>
                <dd>{ROLE_LABELS[identity?.user.role ?? ""] ?? identity?.user.role}</dd>
                <dt>Last sign-in</dt>
                <dd>
                  {identity?.user.last_login_at
                    ? new Date(identity.user.last_login_at).toLocaleString()
                    : "Current session"}
                </dd>
                <dt>Session type</dt>
                <dd>Server-side session — HttpOnly cookie</dd>
                <dt>Session lifecycle</dt>
                <dd>Revoked on sign-out; absolute expiry enforced server-side</dd>
              </dl>
              <p className="note" style={{ marginTop: 12 }}>
                The session is a server-side record; the browser holds only an
                HttpOnly cookie it cannot read. Signing out revokes the session in
                the database, so a copied cookie authenticates nothing afterwards.
              </p>
              <div style={{ marginTop: 16 }}>
                <button className="btn" onClick={() => void logout()} style={{ color: "var(--critical)" }}>
                  Sign out
                </button>
              </div>
            </div>
          </section>
        </>
      )}

      {/* ── ABOUT TAB ────────────────────────────────── */}
      {activeTab === "about" && (
        <>
          <section className="panel">
            <div className="panel-head"><h2>System</h2></div>
            <div className="panel-body">
              <dl className="kv">
                <dt>Application</dt>
                <dd>ObsidianChain Investigation Console</dd>
                <dt>API endpoint</dt>
                <dd className="mono">/api</dd>
                <dt>Stack</dt>
                <dd>React 18 · TypeScript · Vite · FastAPI · SQLite</dd>
                <dt>Network mode</dt>
                <dd>
                  <span style={{ color: "var(--model)" }}>■</span>{" "}
                  Air-gapped — no external resources fetched
                </dd>
              </dl>
            </div>
          </section>

          <section className="panel">
            <div className="panel-head"><h2>Architecture</h2></div>
            <div className="panel-body">
              <p className="note" style={{ margin: 0 }}>
                Analytical results — clustering, risk scores, explanations and
                evidence — are produced by the offline pipeline and read from
                immutable artifacts. This console records investigator state
                beside them and never writes to them. No external resource is
                fetched at any point; the application runs with the network
                disabled.
              </p>
              <div className="settings-arch-items" style={{ marginTop: 16 }}>
                <div className="settings-arch-item">
                  <span className="settings-arch-icon" style={{ color: "var(--model)" }}>◈</span>
                  <div>
                    <strong style={{ fontSize: 12 }}>Analytical Engine</strong>
                    <p className="note" style={{ margin: 0 }}>
                      Random Forest model with 17-stage pipeline. SHAP-based explanations.
                    </p>
                  </div>
                </div>
                <div className="settings-arch-item">
                  <span className="settings-arch-icon" style={{ color: "var(--blockchain)" }}>◈</span>
                  <div>
                    <strong style={{ fontSize: 12 }}>Blockchain Evidence</strong>
                    <p className="note" style={{ margin: 0 }}>
                      On-chain transaction and co-spend analysis. Peel-chain detection.
                    </p>
                  </div>
                </div>
                <div className="settings-arch-item">
                  <span className="settings-arch-icon" style={{ color: "var(--network)" }}>◈</span>
                  <div>
                    <strong style={{ fontSize: 12 }}>Network Evidence</strong>
                    <p className="note" style={{ margin: 0 }}>
                      P2P peer observation correlation. Chi-squared separation testing.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="panel">
            <div className="panel-head"><h2>Problem Statement</h2></div>
            <div className="panel-body">
              <dl className="kv">
                <dt>Competition</dt>
                <dd>Smart India Hackathon 2026</dd>
                <dt>Statement ID</dt>
                <dd className="mono">26146</dd>
                <dt>Domain</dt>
                <dd>Blockchain Forensics · Cryptocurrency Investigation</dd>
              </dl>
            </div>
          </section>
        </>
      )}
    </>
  );
}
