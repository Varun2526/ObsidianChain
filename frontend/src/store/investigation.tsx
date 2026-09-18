/**
 * Loading one investigation, and refusing to render a workspace without it.
 *
 * The defect this replaces
 * ------------------------
 * Five of the eight case-scoped routes read `invId` from the URL, used it
 * only to build a link, and rendered the full GLOBAL alert queue - so
 * `/inv/ANYTHING/alerts` showed a populated investigation workspace for a
 * case that did not exist. An investigation id in a URL is not authorisation
 * and is not even evidence that the case is real.
 *
 * Every case-scoped page now goes through `<CaseGate>`, which loads the case
 * from the backend first and renders one of four things: a loading state, a
 * proper not-found, a proper access-denied, or the page. The backend is what
 * decides which - this component only draws the answer.
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { Link } from "react-router-dom";

import * as api from "../api/console";
import type { Investigation } from "../api/types";

interface CaseContextValue {
  investigation: Investigation;
  reload: () => Promise<void>;
}

const CaseContext = createContext<CaseContextValue | null>(null);

/** The loaded case. Only callable inside `<CaseGate>`, by construction. */
export function useCase(): CaseContextValue {
  const ctx = useContext(CaseContext);
  if (!ctx) throw new Error("useCase must be inside <CaseGate>");
  return ctx;
}

export function useInvestigation(id: string | undefined) {
  const [investigation, setInvestigation] = useState<Investigation | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(
    async (signal?: AbortSignal) => {
      if (!id) {
        setLoading(false);
        return;
      }
      setLoading(true);
      setError(null);
      try {
        setInvestigation(await api.getInvestigation(id, signal));
      } catch (cause) {
        if ((cause as Error)?.name === "AbortError") return;
        setError(cause);
        setInvestigation(null);
      } finally {
        setLoading(false);
      }
    },
    [id],
  );

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  return { investigation, error, loading, reload: () => load() };
}

export function CaseGate({
  invId,
  children,
}: {
  invId: string | undefined;
  children: ReactNode;
}) {
  const { investigation, error, loading, reload } = useInvestigation(invId);

  if (loading) {
    return (
      <section className="panel">
        <div className="panel-body">
          <p className="muted">Loading investigation…</p>
        </div>
      </section>
    );
  }

  if (api.isForbidden(error)) {
    return (
      <StateCard
        title="Access denied"
        tone="denied"
        body={
          "This investigation belongs to another investigator. The backend " +
          "refused the request; nothing about the case is available here."
        }
      />
    );
  }

  if (error || !investigation) {
    return (
      <StateCard
        title="Investigation not found"
        tone="missing"
        body={
          api.isNotFound(error)
            ? "No investigation with this identifier exists."
            : "The investigation could not be loaded. Check that the API is running."
        }
        onRetry={reload}
      />
    );
  }

  return (
    <CaseContext.Provider value={{ investigation, reload }}>
      {children}
    </CaseContext.Provider>
  );
}

function StateCard({
  title,
  body,
  tone,
  onRetry,
}: {
  title: string;
  body: string;
  tone: "denied" | "missing";
  onRetry?: () => void;
}) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>{title}</h2>
        <span className={`status-badge status-${tone === "denied" ? "closed" : "draft"}`}>
          {tone === "denied" ? "403" : "404"}
        </span>
      </div>
      <div className="panel-body">
        <p className="muted" style={{ marginTop: 0 }}>{body}</p>
        <div className="form-actions">
          <Link to="/investigations" className="btn">← All investigations</Link>
          {onRetry ? (
            <button className="btn btn-ghost" onClick={onRetry}>Retry</button>
          ) : null}
        </div>
      </div>
    </section>
  );
}
