/**
 * Whether this deployment has the Elliptic++ reference analysis installed.
 *
 * A server can run without it (an evaluation deployment ships the model and
 * the casework, not the reference dataset). Pages that only show reference
 * data are hidden then, instead of failing with a missing-artifact error.
 * Asked once per page load; null while unknown.
 */
import { useEffect, useState } from "react";

import * as api from "../api/console";

let cached: Promise<boolean> | null = null;

export function useReferenceData(): boolean | null {
  const [present, setPresent] = useState<boolean | null>(null);
  useEffect(() => {
    let live = true;
    cached ??= api.health().then((h) => h.artifacts_present).catch(() => true);
    cached.then((v) => { if (live) setPresent(v); });
    return () => { live = false; };
  }, []);
  return present;
}
