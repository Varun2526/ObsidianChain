/**
 * Admin Users & RBAC Management
 *
 * Provides user provisioning and authorization control:
 * - User account table with roles (ADMIN, INVESTIGATOR, REVIEWER)
 * - User creation modal / form
 * - Deactivation / activation toggles
 */
import { useEffect, useState } from "react";

import * as api from "../../api/console";
import type { UserAccount } from "../../api/types";
import { Skeleton } from "../../components/ui/primitives";
import JellyRadio from "../../components/ui/JellyRadio";

export function AdminUsersPage() {
  const [users, setUsers] = useState<UserAccount[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [roleFilter, setRoleFilter] = useState<string>("ALL");

  // New user form state
  const [username, setUsername] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<"INVESTIGATOR" | "REVIEWER" | "ADMIN">("INVESTIGATOR");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadUsers = () => {
    setLoading(true);
    api.listUsers()
      .then((r) => setUsers(r.users))
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadUsers();
  }, []);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !displayName.trim() || !password.trim()) {
      setError("All fields are mandatory.");
      return;
    }

    setBusy(true);
    setError(null);
    try {
      await api.createUser({
        username: username.trim(),
        display_name: displayName.trim(),
        password,
        role,
      });

      setUsername("");
      setDisplayName("");
      setPassword("");
      setShowCreateModal(false);
      loadUsers();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  };

  const handleDeactivate = async (userId: string) => {
    if (!confirm("Are you sure you want to deactivate this account?")) return;
    try {
      await api.deactivateUser(userId);
      loadUsers();
    } catch (cause) {
      alert(`Deactivation failed: ${cause instanceof Error ? cause.message : String(cause)}`);
    }
  };

  const handleActivate = async (userId: string) => {
    try {
      await api.activateUser(userId);
      loadUsers();
    } catch (cause) {
      alert(`Activation failed: ${cause instanceof Error ? cause.message : String(cause)}`);
    }
  };

  const handleRoleChange = async (userId: string, currentRole: string) => {
    const newRole = prompt(
      `Enter new role for user (INVESTIGATOR, REVIEWER, or ADMIN):`,
      currentRole,
    );
    if (!newRole || newRole.trim().toUpperCase() === currentRole) return;
    const cleaned = newRole.trim().toUpperCase();
    if (!["INVESTIGATOR", "REVIEWER", "ADMIN"].includes(cleaned)) {
      alert("Invalid role. Must be INVESTIGATOR, REVIEWER, or ADMIN.");
      return;
    }
    try {
      await api.setUserRole(userId, cleaned);
      loadUsers();
    } catch (cause) {
      alert(`Role change failed: ${cause instanceof Error ? cause.message : String(cause)}`);
    }
  };

  const handleResetPassword = async (userId: string, userDisplayName: string) => {
    const newPass = prompt(`Enter new password for ${userDisplayName}:`);
    if (!newPass) return;
    if (newPass.length < 8) {
      alert("Password must be at least 8 characters long.");
      return;
    }
    try {
      await api.resetUserPassword(userId, newPass);
      alert(`Password successfully updated for ${userDisplayName}.`);
    } catch (cause) {
      alert(`Password reset failed: ${cause instanceof Error ? cause.message : String(cause)}`);
    }
  };

  const handleDeleteUser = async (userId: string, userName: string) => {
    const confirmText = prompt(
      `EXCEPTIONAL ACTION: Permanently remove user account '${userName}'?\n` +
      `Historical audit records and notes will preserve historical attribution.\n` +
      `Type '${userName}' to confirm deletion:`,
    );
    if (confirmText !== userName) {
      if (confirmText !== null) alert("Confirmation mismatch. Account deletion cancelled.");
      return;
    }
    try {
      await api.deleteUser(userId);
      loadUsers();
    } catch (cause) {
      alert(`Delete user failed: ${cause instanceof Error ? cause.message : String(cause)}`);
    }
  };

  return (
    <>
      <div className="page-header" style={{ marginBottom: 20 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
            <h1 style={{ margin: 0 }}>Users & Access Control</h1>
            <span className="status-badge status-draft">RBAC AUTHORIZATION</span>
          </div>
          <p className="muted" style={{ margin: 0 }}>
            Provision analyst credentials, assign workstation roles, and enforce separation of duties
          </p>
        </div>
        <div>
          <button
            className="btn btn-primary"
            onClick={() => setShowCreateModal(true)}
          >
            + Create New User
          </button>
        </div>
      </div>

      {/* Role Definitions Information Banner */}
      <div className="panel" style={{ marginBottom: 24 }}>
        <div className="panel-body">
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 16 }}>
            <div>
              <strong style={{ color: "var(--text)", display: "block", marginBottom: 4 }}>
                INVESTIGATOR ROLE
              </strong>
              <p className="small muted" style={{ margin: 0 }}>
                Can create cases, ingest and validate data, execute analysis runs, explore graphs, triage alerts, record dispositions, and draft case reports.
              </p>
            </div>
            <div>
              <strong style={{ color: "var(--high)", display: "block", marginBottom: 4 }}>
                REVIEWER ROLE
              </strong>
              <p className="small muted" style={{ margin: 0 }}>
                Independent quality assurance and governance. Can review submitted case work products, approve findings, or return cases for clarification.
              </p>
            </div>
            <div>
              <strong style={{ color: "var(--cyan)", display: "block", marginBottom: 4 }}>
                ADMINISTRATOR ROLE
              </strong>
              <p className="small muted" style={{ margin: 0 }}>
                Workstation configuration, user account provisioning, dataset registry management, and global audit log inspection.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Role Filter & User Accounts Table */}
      <div style={{ marginBottom: 16 }}>
        <JellyRadio
          items={[
            { value: "ALL", label: `All Accounts (${users.length})` },
            { value: "INVESTIGATOR", label: `Investigators (${users.filter((u) => u.role === "INVESTIGATOR").length})` },
            { value: "REVIEWER", label: `Reviewers (${users.filter((u) => u.role === "REVIEWER").length})` },
            { value: "ADMIN", label: `Administrators (${users.filter((u) => u.role === "ADMIN").length})` },
          ]}
          value={roleFilter}
          onChange={(val) => setRoleFilter(val)}
          size="sm"
          chipColor="var(--bg-raised)"
          activeColor="var(--blockchain)"
          textColor="var(--text-dim)"
          activeTextColor="#000000"
          radius={6}
          ariaLabel="Filter users by role"
        />
      </div>

      <section className="panel">
        <div className="panel-head">
          <h2>Active Accounts</h2>
          <span className="small muted">
            {users.filter((u) => roleFilter === "ALL" || u.role === roleFilter).length} Displayed
          </span>
        </div>
        <div className="panel-body flush">
          {loading ? (
            <div style={{ padding: 16 }}><Skeleton rows={5} /></div>
          ) : users.filter((u) => roleFilter === "ALL" || u.role === roleFilter).length === 0 ? (
            <p className="muted" style={{ padding: 16 }}>No users found matching this role.</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Username</th>
                  <th>Display Name</th>
                  <th>Authorized Role</th>
                  <th>Status</th>
                  <th>Created</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {users
                  .filter((u) => roleFilter === "ALL" || u.role === roleFilter)
                  .map((u) => (
                  <tr key={u.id}>
                    <td className="mono font-semibold">{u.username}</td>
                    <td>{u.display_name}</td>
                    <td>
                      <span className={`status-badge ${
                        u.role === "ADMIN" ? "status-closed" :
                        u.role === "REVIEWER" ? "status-review" : "status-active"
                      }`}>
                        {u.role}
                      </span>
                    </td>
                    <td>
                      <span className={`status-badge ${u.active ? "status-active" : "status-closed"}`}>
                        {u.active ? "ACTIVE" : "DEACTIVATED"}
                      </span>
                    </td>
                    <td className="small muted mono">
                      {new Date(u.created_at).toLocaleDateString()}
                    </td>
                    <td>
                      <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                        {u.active ? (
                          <button
                            className="btn btn-sm"
                            style={{ color: "var(--critical)" }}
                            onClick={() => handleDeactivate(u.id)}
                            title="Revoke active sessions and disable access"
                          >
                            Deactivate
                          </button>
                        ) : (
                          <button
                            className="btn btn-sm"
                            style={{ color: "var(--model)" }}
                            onClick={() => handleActivate(u.id)}
                            title="Reactivate account"
                          >
                            Activate
                          </button>
                        )}
                        <button
                          className="btn btn-sm"
                          onClick={() => handleRoleChange(u.id, u.role)}
                          title="Change user's authorization role"
                        >
                          Role
                        </button>
                        <button
                          className="btn btn-sm"
                          onClick={() => handleResetPassword(u.id, u.display_name)}
                          title="Reset user's password"
                        >
                          Reset PW
                        </button>
                        <button
                          className="btn btn-sm"
                          style={{ color: "var(--text-dim)", opacity: 0.7 }}
                          onClick={() => handleDeleteUser(u.id, u.username)}
                          title="Exceptional permanent deletion (preserves audit records)"
                        >
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </section>

      {/* Create User Modal */}
      {showCreateModal && (
        <div className="modal-backdrop" onClick={() => setShowCreateModal(false)}>
          <div className="modal-card" role="dialog" aria-modal="true" aria-label="Create user" style={{ maxWidth: 460 }} onClick={(e) => e.stopPropagation()}>
            <div className="panel-head">
              <h2>Provision New Account</h2>
              <button type="button" className="btn btn-sm btn-ghost" aria-label="Close" onClick={() => setShowCreateModal(false)}>Close</button>
            </div>
            <div className="panel-body">
              <form onSubmit={handleCreate}>
                <div className="form-group">
                  <label htmlFor="modal-username">Username</label>
                  <input
                    id="modal-username"
                    type="text"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    placeholder="e.g. analyst_smith"
                    required
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="modal-displayname">Full Display Name</label>
                  <input
                    id="modal-displayname"
                    type="text"
                    value={displayName}
                    onChange={(e) => setDisplayName(e.target.value)}
                    placeholder="e.g. Special Agent J. Smith"
                    required
                  />
                </div>

                <div className="form-group">
                  <label style={{ display: "block", marginBottom: 8 }}>Workstation Role</label>
                  <JellyRadio
                    items={[
                      { value: "INVESTIGATOR", label: "Investigator" },
                      { value: "REVIEWER", label: "Reviewer" },
                      { value: "ADMIN", label: "Administrator" },
                    ]}
                    value={role}
                    onChange={(val) => setRole(val as any)}
                    size="sm"
                    chipColor="var(--bg-panel)"
                    activeColor="var(--blockchain)"
                    textColor="var(--text-dim)"
                    activeTextColor="#000000"
                    radius={6}
                    ariaLabel="Workstation role selection"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="modal-password">Initial Password</label>
                  <input
                    id="modal-password"
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Enter strong password…"
                    required
                  />
                </div>

                {error && (
                  <div className="banner banner-error" style={{ marginBottom: 14 }}>
                    <p style={{ margin: 0 }}>{error}</p>
                  </div>
                )}

                <div className="form-actions" style={{ marginTop: 20 }}>
                  <button
                    type="button"
                    className="btn"
                    onClick={() => setShowCreateModal(false)}
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={busy}
                  >
                    {busy ? "Provisioning…" : "Create Account →"}
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
