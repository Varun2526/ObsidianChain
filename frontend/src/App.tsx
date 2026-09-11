import { NavLink, Route, Routes } from "react-router-dom";

import { AlertQueue } from "./components/AlertQueue";
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
            Alert queue
          </NavLink>
        </nav>
        <span className="spacer" />
        <span className="small faint">
          Synthetic network data · offline prototype
        </span>
      </header>
      <main className="content">
        <Routes>
          <Route path="/" element={<AlertQueue />} />
          <Route path="/alerts/:alertId" element={<AlertDetailPage />} />
          <Route path="*" element={<AlertQueue />} />
        </Routes>
      </main>
    </div>
  );
}
