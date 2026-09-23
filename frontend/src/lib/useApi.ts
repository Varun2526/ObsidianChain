import { useCallback, useEffect, useRef, useState } from "react";

export interface ApiState<T> {
  data: T | null;
  error: unknown;
  loading: boolean;
  reload: () => void;
}

/**
 * One request tied to a component's lifetime: aborted when the inputs change
 * or the component leaves, so a slow answer for the previous address never
 * lands on the page for the next one.
 */
export function useApi<T>(load: ((signal: AbortSignal) => Promise<T>) | null, deps: unknown[]): ApiState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState<boolean>(load !== null);
  const [tick, setTick] = useState(0);
  const loadRef = useRef(load);
  loadRef.current = load;

  useEffect(() => {
    const fn = loadRef.current;
    if (!fn) { setLoading(false); return; }
    const ctrl = new AbortController();
    setLoading(true);
    setError(null);
    fn(ctrl.signal)
      .then((d) => { if (!ctrl.signal.aborted) { setData(d); setLoading(false); } })
      .catch((e: unknown) => {
        if (ctrl.signal.aborted || (e as Error)?.name === "AbortError") return;
        setError(e); setLoading(false);
      });
    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, reload };
}
