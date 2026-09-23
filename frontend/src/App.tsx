import { lazy } from "react";
import { Navigate, Route, Routes, useParams } from "react-router-dom";
import { useAuth } from "./store/auth";
import { CaseGate } from "./store/investigation";

// Layout
import { AppShell } from "./components/layout/AppShell";

// Global Pages
import { LoginPage } from "./pages/LoginPage";
import { HomePage } from "./pages/HomePage";
const SettingsPage = lazy(() => import("./pages/SettingsPage").then((m) => ({ default: m.SettingsPage })));
const EvaluationPage = lazy(() => import("./pages/EvaluationPage").then((m) => ({ default: m.EvaluationPage })));

// Intelligence pages (chain index, trace, models)
const EntityPage = lazy(() => import("./pages/intel/EntityPage").then((m) => ({ default: m.EntityPage })));
const TransactionPage = lazy(() => import("./pages/intel/TransactionPage").then((m) => ({ default: m.TransactionPage })));
const GraphExplorer = lazy(() => import("./pages/intel/GraphExplorer").then((m) => ({ default: m.GraphExplorer })));
const ModelsPage = lazy(() => import("./pages/intel/ModelsPage").then((m) => ({ default: m.ModelsPage })));

// Investigation Pages
const InvestigationsPage = lazy(() => import("./pages/investigation/InvestigationsPage").then((m) => ({ default: m.InvestigationsPage })));
const NewInvestigation = lazy(() => import("./pages/investigation/NewInvestigation").then((m) => ({ default: m.NewInvestigation })));
const InvestigationOverview = lazy(() => import("./pages/investigation/InvestigationOverview").then((m) => ({ default: m.InvestigationOverview })));
const AlertQueue = lazy(() => import("./pages/investigation/AlertQueue").then((m) => ({ default: m.AlertQueue })));
const AlertDetailPage = lazy(() => import("./pages/investigation/AlertDetail").then((m) => ({ default: m.AlertDetailPage })));
const GraphSubPage = lazy(() => import("./pages/investigation/InvestigationSubPages").then((m) => ({ default: m.GraphSubPage })));
const TimelineSubPage = lazy(() => import("./pages/investigation/InvestigationSubPages").then((m) => ({ default: m.TimelineSubPage })));
const NetworkSubPage = lazy(() => import("./pages/investigation/InvestigationSubPages").then((m) => ({ default: m.NetworkSubPage })));
const EvidenceSubPage = lazy(() => import("./pages/investigation/InvestigationSubPages").then((m) => ({ default: m.EvidenceSubPage })));
const NotesSubPage = lazy(() => import("./pages/investigation/InvestigationSubPages").then((m) => ({ default: m.NotesSubPage })));
const ReportPage = lazy(() => import("./pages/investigation/ReportPage").then((m) => ({ default: m.ReportPage })));
const HistoryPage = lazy(() => import("./pages/investigation/HistoryPage").then((m) => ({ default: m.HistoryPage })));
const InvestigationReview = lazy(() => import("./pages/investigation/InvestigationReview").then((m) => ({ default: m.InvestigationReview })));

// Reviewer Pages
const ReviewerQueue = lazy(() => import("./pages/reviewer/ReviewerQueue").then((m) => ({ default: m.ReviewerQueue })));

// Admin Pages
const AdminDashboard = lazy(() => import("./pages/admin/AdminDashboard").then((m) => ({ default: m.AdminDashboard })));
const AdminUsersPage = lazy(() => import("./pages/admin/AdminUsersPage").then((m) => ({ default: m.AdminUsersPage })));
const AdminDatasetsPage = lazy(() => import("./pages/admin/AdminDatasetsPage").then((m) => ({ default: m.AdminDatasetsPage })));
const AdminAuditLogPage = lazy(() => import("./pages/admin/AdminAuditLogPage").then((m) => ({ default: m.AdminAuditLogPage })));

/**
 * Gate on the SERVER's answer, not on a value the browser wrote itself.
 *
 * `loading` is a distinct third state and must not be collapsed into
 * "logged out": redirecting while `/api/auth/me` is still in flight would
 * bounce every authenticated user to /login on a hard refresh.
 */
function RequireAuth({ children }: { children: React.ReactNode }) {
  const { identity, loading } = useAuth();
  if (loading) return <BootSplash />;
  if (!identity) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function BootSplash() {
  return (
    <div className="boot-splash" role="status">
      <span className="wordmark">Obsidian<b>Chain</b></span>
      <p className="muted small">Restoring session…</p>
    </div>
  );
}

/**
 * Wraps a case-scoped page so it cannot render before the case is loaded.
 *
 * This is presentation, not protection - the backend refuses the underlying
 * data regardless. What it buys is that an unauthorised or mistyped case id
 * shows a 403 or a 404 instead of a workspace shell with global data in it.
 */
function CaseRoute({ children }: { children: React.ReactNode }) {
  const { invId } = useParams();
  return <CaseGate invId={invId}>{children}</CaseGate>;
}

export function App() {
  const { identity, loading } = useAuth();

  return (
    <Routes>
      <Route
        path="/login"
        element={
          loading ? <BootSplash />
            : identity ? <Navigate to="/" replace />
            : <LoginPage />
        }
      />
      <Route
        element={
          <RequireAuth>
            <AppShell />
          </RequireAuth>
        }
      >
        {/* Global Investigator pages */}
        <Route index element={<HomePage />} />
        <Route path="alerts" element={<AlertQueue />} />
        <Route path="alerts/:alertId" element={<AlertDetailPage />} />
        <Route path="investigations" element={<InvestigationsPage />} />
        <Route path="investigations/new" element={<NewInvestigation />} />
        <Route path="settings" element={<SettingsPage />} />
        <Route path="evaluation" element={<EvaluationPage />} />
        <Route path="models" element={<ModelsPage />} />
        <Route path="graph" element={<GraphExplorer />} />
        <Route path="entity/:address" element={<EntityPage />} />
        <Route path="tx/:txid" element={<TransactionPage />} />

        {/* Reviewer Workspace & Queue */}
        <Route path="reviewer" element={<ReviewerQueue />} />

        {/* Admin Workspace */}
        <Route path="admin" element={<AdminDashboard />} />
        <Route path="admin/users" element={<AdminUsersPage />} />
        <Route path="admin/datasets" element={<AdminDatasetsPage />} />
        <Route path="admin/audit" element={<AdminAuditLogPage />} />

        {/* Investigation-scoped pages. Every one loads the case first. */}
        <Route path="inv/:invId" element={<CaseRoute><InvestigationOverview /></CaseRoute>} />
        <Route path="inv/:invId/alerts" element={<CaseRoute><AlertQueue /></CaseRoute>} />
        <Route path="inv/:invId/alerts/:alertId" element={<CaseRoute><AlertDetailPage /></CaseRoute>} />
        <Route path="inv/:invId/graph" element={<CaseRoute><GraphSubPage /></CaseRoute>} />
        <Route path="inv/:invId/timeline" element={<CaseRoute><TimelineSubPage /></CaseRoute>} />
        <Route path="inv/:invId/network" element={<CaseRoute><NetworkSubPage /></CaseRoute>} />
        <Route path="inv/:invId/evidence" element={<CaseRoute><EvidenceSubPage /></CaseRoute>} />
        <Route path="inv/:invId/notes" element={<CaseRoute><NotesSubPage /></CaseRoute>} />
        <Route path="inv/:invId/report" element={<CaseRoute><ReportPage /></CaseRoute>} />
        <Route path="inv/:invId/review" element={<CaseRoute><InvestigationReview /></CaseRoute>} />
        <Route path="inv/:invId/history" element={<CaseRoute><HistoryPage /></CaseRoute>} />

        {/* Catch-all */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
