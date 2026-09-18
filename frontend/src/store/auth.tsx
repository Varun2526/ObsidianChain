/**
 * Authentication state, backed by the server.
 *
 * What changed, and why it matters
 * --------------------------------
 * The previous version of this file accepted any non-empty string as an
 * investigator id, ignored the password entirely (the parameter was named
 * `_password`), and wrote the resulting "session" to localStorage. That was
 * not authentication: two lines in a devtools console impersonated anyone,
 * and the browser was the only thing that ever checked.
 *
 * Now the password is verified by the backend against an scrypt hash, the
 * session is a row in SQLite, and the browser holds an HttpOnly cookie this
 * code cannot read. Nothing about the identity is stored client-side. On
 * startup we ask the server who we are; if it says nobody, we are nobody.
 *
 * `capabilities` is cached for RENDERING only - so a control the backend
 * would refuse can be hidden rather than offered. It is never the security
 * boundary. Every protected operation is re-checked server-side against the
 * session's own role, so editing this value in memory changes what is drawn
 * and nothing else.
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";

import { onSessionLost } from "../api/client";
import * as console_api from "../api/console";
import type { Identity, Role } from "../api/types";

interface AuthContextValue {
  identity: Identity | null;
  /** True until the first /api/auth/me settles. Routes must not decide before. */
  loading: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refresh: () => Promise<void>;
  /** Rendering convenience. Never an authorisation decision. */
  can: (capability: string) => boolean;
  role: Role | null;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [identity, setIdentity] = useState<Identity | null>(null);
  const [loading, setLoading] = useState(true);

  const bootstrap = useCallback(async (signal?: AbortSignal) => {
    try {
      setIdentity(await console_api.me(signal));
    } catch (cause) {
      if ((cause as Error)?.name === "AbortError") return;
      // A 401 here is the ordinary "not logged in" answer, not an error to
      // surface. Anything else - the API being down, for instance - also
      // leaves us unauthenticated, which is the safe state to fail into.
      setIdentity(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void bootstrap(controller.signal);
    return () => controller.abort();
  }, [bootstrap]);

  // A session can lapse between renders - it has an absolute expiry, and an
  // administrator can revoke it. Whichever call discovers that, the whole
  // application finds out here: clearing the identity sends the router to
  // /login rather than leaving an authenticated-looking shell whose every
  // panel fails on its own.
  useEffect(
    () => onSessionLost(() => { setIdentity(null); setLoading(false); }),
    [],
  );

  const login = useCallback(async (username: string, password: string) => {
    // Throws ApiError(401) on bad credentials; the login page renders the
    // backend's own message rather than inventing one.
    setIdentity(await console_api.login(username, password));
  }, []);

  const logout = useCallback(async () => {
    try {
      await console_api.logout();
    } finally {
      // Cleared locally even if the call failed. The server-side revoke is
      // what actually ends the session; this only stops the UI from
      // continuing to draw as though someone were signed in.
      setIdentity(null);
    }
  }, []);

  const can = useCallback(
    (capability: string) => !!identity?.capabilities.includes(capability),
    [identity],
  );

  return (
    <AuthContext.Provider
      value={{
        identity,
        loading,
        login,
        logout,
        refresh: () => bootstrap(),
        can,
        role: identity?.user.role ?? null,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be inside <AuthProvider>");
  return ctx;
}
