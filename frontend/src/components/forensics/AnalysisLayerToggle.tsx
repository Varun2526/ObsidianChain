/**
 * CHAIN / NETWORK / FUSED — a view mode, not three models.
 *
 * What it does and does not mean
 * -----------------------------
 * One analytical run produced everything the alert page shows. The toggle
 * chooses which of that already-fetched evidence is displayed. It triggers no
 * request, no scoring and no recomputation, and it must never be read as
 * "the chain model said X and the network model said Y" - there is one
 * model, and the network layer is not a model at all.
 *
 * Why the mode lives in the URL
 * -----------------------------
 * A search parameter, so the mode survives a reload and can be sent to a
 * colleague with the alert. It is a rendering preference and nothing more:
 * the backend neither reads it nor cares, and no authorisation depends on
 * it.
 *
 * Switching preserves the investigator's place
 * --------------------------------------------
 * The route, the case and the selected alert are untouched - only a query
 * parameter changes - so CHAIN -> NETWORK -> FUSED never loses the work in
 * front of someone.
 */
import { useCallback } from "react";
import { useSearchParams } from "react-router-dom";

import type { AnalysisLayer } from "../../api/types";

const LAYERS: { id: AnalysisLayer; label: string; hint: string }[] = [
  {
    id: "CHAIN",
    label: "Chain",
    hint: "Blockchain evidence only: behaviour, graph, chain structure, "
        + "mixing-like structure, model risk and its explanation.",
  },
  {
    id: "NETWORK",
    label: "Network",
    hint: "Network evidence only: announcement observations, propagation "
        + "context and separation evidence. It never attributes ownership.",
  },
  {
    id: "FUSED",
    label: "Fused",
    hint: "Both, with each source labelled.",
  },
];

export const DEFAULT_LAYER: AnalysisLayer = "FUSED";

function parseLayer(raw: string | null): AnalysisLayer {
  const upper = (raw ?? "").toUpperCase();
  return upper === "CHAIN" || upper === "NETWORK" || upper === "FUSED"
    ? upper
    : DEFAULT_LAYER;
}

/** The current layer, and a setter that changes nothing else about the route. */
export function useAnalysisLayer(): [AnalysisLayer, (next: AnalysisLayer) => void] {
  const [params, setParams] = useSearchParams();
  const layer = parseLayer(params.get("layer"));

  const setLayer = useCallback(
    (next: AnalysisLayer) => {
      const updated = new URLSearchParams(params);
      updated.set("layer", next.toLowerCase());
      // `replace` so flipping between layers does not fill the back button
      // with view changes the investigator would have to click through to
      // get back to where they came from.
      setParams(updated, { replace: true });
    },
    [params, setParams],
  );

  return [layer, setLayer];
}

export function AnalysisLayerToggle({
  layer,
  onChange,
  networkAvailable = true,
}: {
  layer: AnalysisLayer;
  onChange: (next: AnalysisLayer) => void;
  /** False when this run has no network observations to show. */
  networkAvailable?: boolean;
}) {
  const active = LAYERS.find((l) => l.id === layer) ?? LAYERS[2]!;
  return (
    <div className="layer-toggle">
      <span className="layer-toggle-label">Analysis layer</span>
      <div className="toggles" role="group" aria-label="Analysis layer">
        {LAYERS.map((entry) => (
          <button
            key={entry.id}
            className="toggle"
            aria-pressed={layer === entry.id}
            onClick={() => onChange(entry.id)}
            title={entry.hint}
          >
            {entry.label}
          </button>
        ))}
      </div>
      <span className="layer-toggle-hint">{active.hint}</span>
      {!networkAvailable && layer !== "CHAIN" && (
        <span className="layer-toggle-warning">
          Network analysis unavailable for this alert — no announcement
          observations were correlated to it.
        </span>
      )}
      <span className="layer-toggle-note">
        A view over one analytical run. Switching selects which evidence is
        shown; nothing is recomputed and there are not three models.
      </span>
    </div>
  );
}
