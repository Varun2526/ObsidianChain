/**
 * Application shell: sidebar + topbar + content area.
 *
 * The case shown in the sidebar is fetched from the backend rather than read
 * out of browser storage, so the chrome cannot claim a case that does not
 * exist or that this user may not see.
 */
import { NavLink, Outlet, useParams, useNavigate } from "react-router-dom";
import { useAuth } from "../store/auth";
import { useInvestigation } from "../store/investigation";

const NAV_ITEMS = [
  { to: "/", icon: "⌂", label: "Home" },
  { to: "/alerts", icon: "▲", label: "Alerts" },
  { to: "/investigations", icon: "◫", label: "Investigations" },
  { to: "/evaluation", icon: "⚗", label: "Evaluation" },
  { to: "/settings", icon: "⚙", label: "Settings" },
];

const INV_NAV = [
  { sub: "", label: "Overview" },
  { sub: "/alerts", label: "Alerts" },
  { sub: "/graph", label: "Graph" },
  { sub: "/timeline", label: "Timeline" },
  { sub: "/network", label: "Network" },
  { sub: "/evidence", label: "Evidence" },
  { sub: "/report", label: "Report" },
  { sub: "/history", label: "History" },
];

export function AppShell() {
  const { identity, logout } = useAuth();
  const { invId } = useParams();
  const navigate = useNavigate();
  // Undefined while loading and when the case is missing or forbidden, so
  // the sub-navigation simply does not appear rather than appearing empty.
  const { investigation: inv } = useInvestigation(invId);

  const signOut = async () => {
    await logout();
    navigate("/login");
  };

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <span className="sidebar-logo" />
          <span className="sidebar-title">OBSIDIANCHAIN</span>
        </div>

        <nav className="sidebar-nav">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) => `sidebar-link${isActive && !inv ? " active" : ""}`}
            >
              <span className="sidebar-icon">{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </nav>

        {inv && (
          <div className="sidebar-inv">
            <div className="sidebar-inv-header">
              <span className="sidebar-inv-id">{inv.case_label}</span>
              <span className="sidebar-inv-name">{inv.name}</span>
            </div>
            <nav className="sidebar-nav">
              {INV_NAV.map((item) => (
                <NavLink
                  key={item.sub}
                  to={`/inv/${inv.id}${item.sub}`}
                  end={item.sub === ""}
                  className={({ isActive }) => `sidebar-link${isActive ? " active" : ""}`}
                >
                  {item.label}
                </NavLink>
              ))}
            </nav>
          </div>
        )}

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
          {inv ? (
            <div className="topbar-inv">
              <span className="topbar-inv-id">{inv.case_label}</span>
              <span className="topbar-sep">·</span>
              <span className="topbar-inv-name">{inv.name}</span>
              <span className={`status-badge status-${inv.status.toLowerCase()}`}>
                {inv.status}
              </span>
            </div>
          ) : (
            <span className="topbar-brand">Investigation Console</span>
          )}
          <span className="spacer" />
          <span className="topbar-user">
            {identity?.user.username}
            <span className="topbar-role">{identity?.user.role}</span>
          </span>
        </header>
        <main className="content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
