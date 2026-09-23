/**
 * Application shell: navigation rail, top bar with search, content area.
 *
 * The rail is grouped by the investigator's workflow (triage, investigate,
 * understand the model, administer). Role only decides which groups are
 * drawn; the backend re-checks every request regardless.
 *
 * The case shown in the chrome is fetched from the backend rather than read
 * out of browser storage, so the chrome cannot claim a case that does not
 * exist or that this user may not see.
 */
import { Suspense, useEffect, useState } from "react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";

import { useAuth } from "../../store/auth";
import { useInvestigation } from "../../store/investigation";
import { CommandPalette } from "../modals/CommandPalette";
import { ErrorBoundary } from "../ui/ErrorBoundary";
import { Icon } from "../ui/Icon";
import type { IconName } from "../ui/Icon";
import { PersistentCaseHeader } from "./CaseChrome";

interface NavItem { to: string; icon: IconName; label: string; end?: boolean }
interface NavGroup { heading: string; items: NavItem[]; roles?: string[] }

const NAV: NavGroup[] = [
  {
    heading: "Investigate",
    items: [
      { to: "/", icon: "overview", label: "Overview", end: true },
      { to: "/alerts", icon: "alert", label: "Alerts" },
      { to: "/investigations", icon: "folder", label: "Investigations" },
      { to: "/graph", icon: "graph", label: "Graph explorer" },
    ],
  },
  {
    heading: "Review",
    roles: ["REVIEWER", "ADMIN"],
    items: [{ to: "/reviewer", icon: "review", label: "Review queue" }],
  },
  {
    heading: "Intelligence",
    items: [
      { to: "/models", icon: "model", label: "Models" },
      { to: "/evaluation", icon: "flask", label: "Synthetic evaluation" },
    ],
  },
  {
    heading: "Administration",
    roles: ["ADMIN"],
    items: [
      { to: "/admin", icon: "shield", label: "System", end: true },
      { to: "/admin/users", icon: "users", label: "Users and roles" },
      { to: "/admin/datasets", icon: "database", label: "Datasets" },
      { to: "/admin/audit", icon: "log", label: "Audit log" },
    ],
  },
];

const FLUSH_ROUTES = [/^\/graph/];

function pageTitle(path: string): string {
  if (path === "/") return "Overview";
  if (path.startsWith("/alerts/")) return "Alert";
  if (path.startsWith("/alerts")) return "Alerts";
  if (path.startsWith("/investigations/new")) return "New investigation";
  if (path.startsWith("/investigations")) return "Investigations";
  if (path.startsWith("/graph")) return "Graph explorer";
  if (path.startsWith("/entity/")) return "Address";
  if (path.startsWith("/tx/")) return "Transaction";
  if (path.startsWith("/models")) return "Models";
  if (path.startsWith("/evaluation")) return "Synthetic evaluation";
  if (path.startsWith("/reviewer")) return "Review queue";
  if (path.startsWith("/admin/users")) return "Users and roles";
  if (path.startsWith("/admin/datasets")) return "Datasets";
  if (path.startsWith("/admin/audit")) return "Audit log";
  if (path.startsWith("/admin")) return "System";
  if (path.startsWith("/settings")) return "Settings";
  return "";
}

export function AppShell() {
  const { identity, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const pathMatch = location.pathname.match(/^\/inv\/([^/]+)/);
  const currentInvId = pathMatch ? pathMatch[1] : undefined;
  const { investigation: inv } = useInvestigation(currentInvId);
  const activeInv = currentInvId ? inv : null;

  const [searchOpen, setSearchOpen] = useState(false);
  const [railOpen, setRailOpen] = useState(false);
  const role = identity?.user.role ?? "INVESTIGATOR";
  const flush = FLUSH_ROUTES.some((r) => r.test(location.pathname));

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setSearchOpen((v) => !v);
        return;
      }
      const t = e.target as HTMLElement | null;
      const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
      if (e.key === "/" && !typing) { e.preventDefault(); setSearchOpen(true); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => { setRailOpen(false); }, [location.pathname]);

  useEffect(() => {
    const t = activeInv ? `${activeInv.case_label} · ${activeInv.name}` : pageTitle(location.pathname);
    document.title = t ? `${t} · ObsidianChain` : "ObsidianChain";
  }, [location.pathname, activeInv]);

  const signOut = async () => {
    await logout();
    navigate("/login");
  };

  return (
    <>
      <a className="skip-link" href="#main">Skip to content</a>
      <CommandPalette open={searchOpen} onClose={() => setSearchOpen(false)} />
      <div className="shell" data-rail-open={railOpen}>
        {railOpen && <div className="rail-scrim" onClick={() => setRailOpen(false)} aria-hidden="true" />}
        <aside className="rail" aria-label="Primary">
          <NavLink to="/" className="rail-brand" aria-label="ObsidianChain overview">
            <span>
              <span className="wordmark">Obsidian<b>Chain</b></span>
              <span className="wordmark-sub">Investigation and risk intelligence</span>
            </span>
          </NavLink>
          <nav className="rail-nav">
            {NAV.filter((g) => !g.roles || g.roles.includes(role)).map((g) => (
              <div className="rail-section" key={g.heading}>
                <div className="rail-heading">{g.heading}</div>
                {g.items.map((item) => (
                  <NavLink key={item.to} to={item.to} end={item.end}
                           className={({ isActive }) => `rail-link${isActive ? " active" : ""}`}>
                    <Icon name={item.icon} />
                    <span>{item.label}</span>
                  </NavLink>
                ))}
              </div>
            ))}
            <div className="rail-section">
              <NavLink to="/settings" className={({ isActive }) => `rail-link${isActive ? " active" : ""}`}>
                <Icon name="settings" /><span>Settings</span>
              </NavLink>
            </div>
          </nav>
          <div className="rail-foot">
            <div className="rail-user">
              <span className="rail-user-name">{identity?.user.display_name ?? identity?.user.username ?? "Signed out"}</span>
              <span className="rail-user-role">{identity?.user.role}</span>
            </div>
            <button type="button" className="btn btn-sm btn-ghost btn-icon" onClick={signOut} aria-label="Sign out" title="Sign out">
              <Icon name="signout" />
            </button>
          </div>
        </aside>

        <div className="shell-main">
          <header className="topbar">
            <button type="button" className="btn btn-sm btn-ghost btn-icon menu-toggle" aria-label="Open navigation"
                    aria-expanded={railOpen} onClick={() => setRailOpen(true)}>
              <Icon name="menu" />
            </button>
            <div className="topbar-context">
              {activeInv ? (
                <>
                  <span className="mono small">{activeInv.case_label}</span>
                  <span className="topbar-sep">/</span>
                  <strong>{activeInv.name}</strong>
                  <span className={`status-badge status-${activeInv.status.toLowerCase()}`}>{activeInv.status}</span>
                </>
              ) : (
                <strong>{pageTitle(location.pathname)}</strong>
              )}
            </div>
            <button type="button" className="topbar-search" onClick={() => setSearchOpen(true)} aria-label="Search (Ctrl K)">
              <Icon name="search" size={14} />
              <span>Search address, transaction, alert, case</span>
              <kbd>⌘K</kbd>
            </button>
            <span className="env-chip" title="Analytical artifacts are read from local, provenance-checked files. No network access.">Elliptic++ · offline</span>
          </header>

          {activeInv && <PersistentCaseHeader inv={activeInv} />}

          <main id="main" className={`content${flush ? " content-flush" : ""}`} tabIndex={-1}>
            <ErrorBoundary resetKey={location.pathname}>
              <Suspense fallback={<div className="panel" aria-busy="true"><div className="panel-body"><div className="skeleton" /><div className="skeleton" style={{ width: "70%" }} /></div></div>}>
                <Outlet />
              </Suspense>
            </ErrorBoundary>
          </main>
        </div>
      </div>
    </>
  );
}
