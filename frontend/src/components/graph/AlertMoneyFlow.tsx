/**
 * An alert's money flow: its members traced over the chain index, with the
 * cluster and its relay peers. Replaces the member-only relationship graph,
 * which could not show where value came from or went.
 */
import { getAlertGraph } from "../../api/intel";
import { useApi } from "../../lib/useApi";
import { ErrorState } from "../ui/ErrorState";
import { Skeleton } from "../ui/primitives";
import { FlowPreview } from "./FlowPreview";

export function AlertMoneyFlow({ alertId }: { alertId: string }) {
  const g = useApi((s) => getAlertGraph(alertId, { hops: 1, maxNodes: 300 }, s), [alertId]);
  if (g.loading) return <section className="panel"><div className="panel-head"><h2>Money flow</h2></div><Skeleton rows={4} /></section>;
  if (g.error) return <ErrorState error={g.error} onRetry={g.reload} />;
  const r = g.data;
  if (!r || !Array.isArray(r.graph?.nodes) || !Array.isArray(r.graph?.edges)) {
    return <section className="panel"><div className="panel-head"><h2>Money flow</h2></div>
      <div className="panel-body"><p className="muted small">The graph endpoint returned no graph for this alert.</p></div></section>;
  }
  return (
    <FlowPreview
      nodes={r.graph.nodes}
      edges={r.graph.edges}
      truncated={r.truncated}
      explorerHref={`/graph?alert=${encodeURIComponent(alertId)}&hops=2`}
      title="Money flow around the cluster"
      height={480}
      note={<>
        {r.seeds?.omitted > 0 ? `Traced from the ${r.seeds.used} highest-scored of ${r.seeds.members_total} members, one hop each way. ` : "One hop each way from every member. "}
        {r.hub_transactions_skipped?.length > 0 && `${r.hub_transactions_skipped.length} transaction(s) with more than ${r.hub_threshold} participants were not crossed. `}
        Membership is a heuristic; relay peers are vantage points, never senders.
      </>}
    />
  );
}
