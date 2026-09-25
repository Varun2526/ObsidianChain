/** The latest COMPLETE uploaded-dataset run of a case, if it has one. */
import * as api from "../../api/console";
import { useApi } from "../../lib/useApi";

export function useCaseRun(invId: string) {
  const inv = useApi((s) => api.getInvestigation(invId, s), [invId]);
  const run = (inv.data?.datasets ?? [])
    .map((d) => ({ dataset: d, run: d.analysis_run }))
    .find((x) => x.run?.status === "COMPLETE");
  return { loading: inv.loading, runId: run?.run?.id ?? null, filename: run?.dataset.filename ?? null };
}
