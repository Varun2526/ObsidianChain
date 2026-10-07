/**
 * Adjustable side panels for the graph workspace.
 *
 * Each panel can be dragged wider or narrower, nudged with the arrow keys,
 * reset with a double-click, or hidden to give the canvas the full width.
 * Widths are remembered per browser. Below the stacked breakpoint (CSS) the
 * handles are not shown and the panels flow vertically as before.
 */
import { useCallback, useEffect, useRef, useState } from "react";

type Side = "left" | "right";

const LIMITS: Record<Side, { min: number; max: number; initial: number }> = {
  left: { min: 200, max: 560, initial: 280 },
  right: { min: 260, max: 680, initial: 340 },
};
const STORAGE_KEY = "oc.graph.panels";

interface PanelState { left: number; right: number; leftHidden: boolean; rightHidden: boolean }

function load(): PanelState {
  const fallback = { left: LIMITS.left.initial, right: LIMITS.right.initial, leftHidden: false, rightHidden: false };
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? { ...fallback, ...JSON.parse(raw) } : fallback;
  } catch {
    return fallback;
  }
}

const clamp = (side: Side, w: number) => Math.round(Math.min(LIMITS[side].max, Math.max(LIMITS[side].min, w)));

/** The canvas never gets narrower than this, however wide the panels are dragged. */
const MIN_CANVAS = 360;

export function usePanelLayout() {
  const [state, setState] = useState<PanelState>(load);
  const container = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    try { window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch { /* storage unavailable */ }
  }, [state]);

  const setWidth = useCallback((side: Side, w: number) =>
    setState((s) => {
      const other: Side = side === "left" ? "right" : "left";
      const total = container.current?.clientWidth ?? Infinity;
      const otherW = s[`${other}Hidden`] ? 0 : s[other];
      const room = total - otherW - 16 - MIN_CANVAS;   // 16 = the two handles
      const capped = Math.min(clamp(side, w), Math.max(LIMITS[side].min, room));
      return { ...s, [side]: Math.round(capped), [`${side}Hidden`]: false };
    }), []);
  const reset = useCallback((side: Side) => setWidth(side, LIMITS[side].initial), [setWidth]);
  const toggle = useCallback((side: Side) =>
    setState((s) => ({ ...s, [`${side}Hidden`]: !s[`${side}Hidden` as const] })), []);

  const style = {
    "--ws-left": state.leftHidden ? "0px" : `${state.left}px`,
    "--ws-right": state.rightHidden ? "0px" : `${state.right}px`,
  } as React.CSSProperties;

  return { state, style, setWidth, reset, toggle, container };
}

export function WorkspaceResizer({ side, width, hidden, onResize, onReset, onToggle }: {
  side: Side; width: number; hidden: boolean;
  onResize: (side: Side, w: number) => void; onReset: (side: Side) => void; onToggle: (side: Side) => void;
}) {
  const start = useRef<{ x: number; w: number } | null>(null);
  const [dragging, setDragging] = useState(false);

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (hidden) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    start.current = { x: e.clientX, w: width };
    setDragging(true);
  };
  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!start.current) return;
    const dx = e.clientX - start.current.x;
    onResize(side, start.current.w + (side === "left" ? dx : -dx));
  };
  const end = () => { start.current = null; setDragging(false); };
  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? 64 : 16;
    const grow = side === "left" ? "ArrowRight" : "ArrowLeft";
    const shrink = side === "left" ? "ArrowLeft" : "ArrowRight";
    if (e.key === grow) { e.preventDefault(); onResize(side, width + step); }
    else if (e.key === shrink) { e.preventDefault(); onResize(side, width - step); }
    else if (e.key === "Enter") { e.preventDefault(); onToggle(side); }
  };

  return (
    <div
      className={`ws-resizer ws-resizer-${side}${dragging ? " dragging" : ""}${hidden ? " is-hidden" : ""}`}
      role="separator"
      aria-orientation="vertical"
      aria-label={`Resize the ${side === "left" ? "trace settings" : "inspector"} panel. Arrow keys resize, Enter hides or shows, double-click resets.`}
      aria-valuenow={hidden ? 0 : width}
      aria-valuemin={LIMITS[side].min}
      aria-valuemax={LIMITS[side].max}
      tabIndex={0}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={end}
      onPointerCancel={end}
      onDoubleClick={() => onReset(side)}
      onKeyDown={onKeyDown}
    >
      <button
        type="button"
        className="ws-resizer-toggle"
        aria-label={`${hidden ? "Show" : "Hide"} the ${side === "left" ? "trace settings" : "inspector"} panel`}
        title={hidden ? "Show panel" : "Hide panel"}
        onPointerDown={(e) => e.stopPropagation()}
        onClick={() => onToggle(side)}
      >
        {(side === "left") !== hidden ? "‹" : "›"}
      </button>
    </div>
  );
}
