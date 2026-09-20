import { Navigate, Route, Routes, useParams } from "react-router-dom";
import { useAuth } from "./store/auth";
import { CaseGate } from "./store/investigation";

// Layout
import { AppShell } from "./components/layout/AppShell";

// Global Pages
import { LoginPage } from "./pages/LoginPage";
import { HomePage } from "./pages/HomePage";
import { SettingsPage } from "./pages/SettingsPage";
import { EvaluationPage } from "./pages/EvaluationPage";

// Investigation Pages
import { InvestigationsPage } from "./pages/investigation/InvestigationsPage";
import { NewInvestigation } from "./pages/investigation/NewInvestigation";
import { InvestigationOverview } from "./pages/investigation/InvestigationOverview";
import { AlertQueue } from "./pages/investigation/AlertQueue";
import { AlertDetailPage } from "./pages/investigation/AlertDetail";
import {
  GraphSubPage,
  TimelineSubPage,
  NetworkSubPage,
  EvidenceSubPage,
  NotesSubPage,
} from "./pages/investigation/InvestigationSubPages";
import { ReportPage } from "./pages/investigation/ReportPage";
import { HistoryPage } from "./pages/investigation/HistoryPage";
import { InvestigationReview } from "./pages/investigation/InvestigationReview";

// Reviewer Pages
import { ReviewerQueue } from "./pages/reviewer/ReviewerQueue";

// Admin Pages
import { AdminDashboard } from "./pages/admin/AdminDashboard";
import { AdminUsersPage } from "./pages/admin/AdminUsersPage";
import { AdminDatasetsPage } from "./pages/admin/AdminDatasetsPage";
import { AdminAuditLogPage } from "./pages/admin/AdminAuditLogPage";

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
    <div className="boot-splash">
      <span className="login-logo" />
      <p className="muted">Restoring session…</p>
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
