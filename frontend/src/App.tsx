import { NavLink, Route, Routes } from "react-router-dom";

import { AlertQueue } from "./components/AlertQueue";
import { Home } from "./components/Home";
import { IngestPage } from "./components/IngestPage";
import { AlertDetailPage } from "./components/AlertDetail";

export function App() {
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-name">OBSIDIANCHAIN</span>
          <span className="brand-sub">Investigation Console</span>
        </div>
        <nav>
          <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : "")}>
            Start
          </NavLink>
          <NavLink to="/ingest" className={({ isActive }) => (isActive ? "active" : "")}>
            Load data
          </NavLink>
          <NavLink to="/alerts" className={({ isActive }) => (isActive ? "active" : "")}>
            Investigate
          </NavLink>
        </nav>
        <span className="spacer" />
        <span className="small faint">
          Synthetic network data · offline prototype
        </span>
      </header>
      <main className="content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/ingest" element={<IngestPage />} />
          <Route path="/alerts" element={<AlertQueue />} />
          <Route path="/alerts/:alertId" element={<AlertDetailPage />} />
          <Route path="*" element={<Home />} />
        </Routes>
      </main>
    </div>
  );
}
