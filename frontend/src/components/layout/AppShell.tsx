/**
 * Application shell: sidebar + topbar + content area.
 *
 * The case shown in the sidebar is fetched from the backend rather than read
 * out of browser storage, so the chrome cannot claim a case that does not
 * exist or that this user may not see.
 */
import { useState, useEffect } from "react";
import { NavLink, Outlet, useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "../../store/auth";
import { useInvestigation } from "../../store/investigation";
import { PersistentCaseHeader } from "./CaseChrome";
import { OmniSearchModal } from "../modals/OmniSearchModal";
import { HookSidebar } from "../ui/hook-sidebar";

const INVESTIGATOR_NAV = [
  { to: "/", icon: "⌂", label: "Home" },
  { to: "/investigations", icon: "◫", label: "Investigations" },
  { to: "/settings", icon: "⚙", label: "Settings" },
];

const REVIEWER_NAV = [
  { to: "/reviewer", icon: "✓", label: "Review Queue" },
  { to: "/investigations", icon: "◫", label: "Investigations" },
  { to: "/settings", icon: "⚙", label: "Settings" },
];

const ADMIN_NAV = [
  { to: "/admin", icon: "❖", label: "System Overview" },
  { to: "/admin/users", icon: "👥", label: "Users & Roles" },
  { to: "/investigations", icon: "◫", label: "Investigations" },
  { to: "/admin/datasets", icon: "🗄", label: "Datasets" },
  { to: "/admin/audit", icon: "📋", label: "Audit Log" },
  { to: "/settings", icon: "⚙", label: "System Settings" },
];

export function AppShell() {
  const { identity, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  // Only consider investigation active when on an /inv/:invId route
  const pathMatch = location.pathname.match(/^\/inv\/([^/]+)/);
  const currentInvId = pathMatch ? pathMatch[1] : undefined;
  const isCaseRoute = Boolean(currentInvId);
  const { investigation: inv } = useInvestigation(currentInvId);
  const activeInv = isCaseRoute ? inv : null;

  const [searchOpen, setSearchOpen] = useState(false);

  const role = identity?.user.role ?? "INVESTIGATOR";

  // Determine active workspace from location
  const inAdmin = location.pathname.startsWith("/admin");
  const inReviewer = location.pathname.startsWith("/reviewer");

  const navItems = inAdmin ? ADMIN_NAV : inReviewer ? REVIEWER_NAV : INVESTIGATOR_NAV;

  // Global keyboard shortcut ⌘K / Ctrl+K
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setSearchOpen((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleGlobalKeyDown);
    return () => window.removeEventListener("keydown", handleGlobalKeyDown);
  }, []);

  const signOut = async () => {
    await logout();
    navigate("/login");
  };

  return (
    <>
      <OmniSearchModal open={searchOpen} onClose={() => setSearchOpen(false)} />
      <div className="shell">

      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="sidebar-monogram">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2L3 7v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-9-5z" />
            </svg>
          </div>
          <span className="sidebar-title">OBSIDIANCHAIN</span>
        </div>

        {/* Global Omni-Search button */}
        <div className="sidebar-search-container">
          <button
            className="sidebar-search-btn"
            onClick={() => setSearchOpen(true)}
            title="Global discovery search (⌘K)"
          >
            <span className="search-icon">🔍</span>
            <span className="search-placeholder">Quick Search…</span>
            <kbd className="search-shortcut">⌘K</kbd>
          </button>
        </div>

        <div style={{ flex: 1, padding: "8px 0" }}>
          <HookSidebar
            items={navItems.map((item) => ({
              label: item.label,
              href: item.to,
              icon: item.icon,
            }))}
            color="var(--cyan, #00f0aa)"
            dashed={true}
          />
        </div>

        <div className="sidebar-footer">
          <span className="sidebar-user">
            {identity?.user.display_name ?? "—"}
            <span className="sidebar-role">{identity?.user.role}</span>
          </span>
          <button className="sidebar-logout" onClick={signOut}>Sign out</button>
        </div>
      </aside>

      <div className="shell-main">
        <header className="topbar">
          {activeInv ? (
            <div className="topbar-inv">
              <span className="topbar-inv-id">{activeInv.case_label}</span>
              <span className="topbar-sep">·</span>
              <span className="topbar-inv-name">{activeInv.name}</span>
              <span className={`status-badge status-${activeInv.status.toLowerCase()}`}>
                {activeInv.status}
              </span>
            </div>
          ) : (
            <div className="topbar-workspace-indicator">
              <span className="workspace-tag">
                {inAdmin ? "ADMIN WORKSPACE" : inReviewer ? "REVIEWER WORKSPACE" : "INVESTIGATOR WORKSPACE"}
              </span>
            </div>
          )}

          <span className="spacer" />

          {/* Authorization-Aware Workspace Switcher */}
          {role === "ADMIN" && (
            <div className="workspace-switcher">
              <NavLink to="/" end className={({ isActive }) => `ws-tab${isActive ? " active" : ""}`}>
                Investigator
              </NavLink>
              <NavLink to="/reviewer" className={({ isActive }) => `ws-tab${isActive ? " active" : ""}`}>
                Reviewer
              </NavLink>
              <NavLink to="/admin" className={({ isActive }) => `ws-tab${isActive ? " active" : ""}`}>
                Admin
              </NavLink>
            </div>
          )}

          {role === "REVIEWER" && (
            <div className="workspace-switcher">
              <NavLink to="/reviewer" className={({ isActive }) => `ws-tab${isActive ? " active" : ""}`}>
                Review Queue
              </NavLink>
              <NavLink to="/investigations" className={({ isActive }) => `ws-tab${isActive ? " active" : ""}`}>
                Cases
              </NavLink>
            </div>
          )}

          <div className="topbar-user-chip">
            <span className="user-dot" />
            <span className="username">{identity?.user.username}</span>
            <span className="role-pill">{identity?.user.role}</span>
          </div>
        </header>

        {activeInv && <PersistentCaseHeader inv={activeInv} />}

        <main className="content">
          <Outlet />
        </main>
      </div>
    </div>
    </>
  );
}

