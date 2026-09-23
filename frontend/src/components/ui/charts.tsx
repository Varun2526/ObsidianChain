import { useEffect, useRef, useState } from "react";

/** The chart's real pixel width, so SVG text is never stretched by a non-uniform viewBox. */
function useWidth(fallback = 600) {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => { if (e && e.contentRect.width > 0) setW(Math.round(e.contentRect.width)); });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

/**
 * Small SVG charts. No charting library: these are a handful of bars and
 * lines, and each one states its axis and unit in text.
 */

export interface BarDatum { x: number; values: number[] }

/** Stacked bars over timesteps (1-49 in Elliptic++). */
export function StepBars({ data, colors, labels, height = 96, domain = [1, 49] as [number, number], ariaLabel }: {
  data: BarDatum[]; colors: string[]; labels: string[]; height?: number; domain?: [number, number]; ariaLabel: string;
}) {
  const [ref, w] = useWidth();
  const pad = { l: 28, r: 6, t: 6, b: 18 };
  const n = domain[1] - domain[0] + 1;
  const bw = (w - pad.l - pad.r) / n;
  const max = Math.max(1, ...data.map((d) => d.values.reduce((a, b) => a + b, 0)));
  const y = (v: number) => (height - pad.b) - (v / max) * (height - pad.t - pad.b);
  const ticks = [domain[0], Math.round((domain[0] + domain[1]) / 2), domain[1]];
  return (
    <figure style={{ margin: 0 }} ref={ref}>
      <svg className="chart" viewBox={`0 0 ${w} ${height}`} role="img" aria-label={ariaLabel} style={{ height }}>
        <line className="axis" x1={pad.l} x2={w - pad.r} y1={height - pad.b} y2={height - pad.b} />
        <text x={pad.l - 4} y={pad.t + 8} textAnchor="end">{max}</text>
        <text x={pad.l - 4} y={height - pad.b} textAnchor="end">0</text>
        {ticks.map((t) => (
          <text key={t} x={pad.l + (t - domain[0] + 0.5) * bw} y={height - 4} textAnchor="middle">t{t}</text>
        ))}
        {data.map((d) => {
          let acc = 0;
          return d.values.map((v, i) => {
            if (v <= 0) return null;
            const top = y(acc + v);
            const bottom = y(acc);
            acc += v;
            return (
              <rect key={`${d.x}-${i}`} x={pad.l + (d.x - domain[0]) * bw + 1} width={Math.max(1, bw - 2)}
                    y={top} height={Math.max(0.5, bottom - top)} fill={colors[i]}>
                <title>t{d.x} {labels[i]}: {v}</title>
              </rect>
            );
          });
        })}
      </svg>
      <figcaption className="chart-legend">
        {labels.map((l, i) => <span key={l}><i style={{ background: colors[i], height: 8 }} />{l}</span>)}
      </figcaption>
    </figure>
  );
}

/** One value per category, 0-1 scale, with an optional reference line. */
export function ScoreBars({ data, height = 140, reference, refLabel, ariaLabel, color = "var(--oc-ev-model)" }: {
  data: { label: string; value: number | null; tone?: string; note?: string }[];
  height?: number; reference?: number; refLabel?: string; ariaLabel: string; color?: string;
}) {
  const [ref, w] = useWidth();
  const pad = { l: 30, r: 6, t: 8, b: 22 };
  const bw = (w - pad.l - pad.r) / Math.max(1, data.length);
  const y = (v: number) => (height - pad.b) - v * (height - pad.t - pad.b);
  return (
    <div ref={ref}>
    <svg className="chart" viewBox={`0 0 ${w} ${height}`} role="img" aria-label={ariaLabel} style={{ height }}>
      {[0, 0.5, 1].map((g) => (
        <g key={g}>
          <line className="grid" x1={pad.l} x2={w - pad.r} y1={y(g)} y2={y(g)} />
          <text x={pad.l - 4} y={y(g) + 3} textAnchor="end">{g.toFixed(1)}</text>
        </g>
      ))}
      {data.map((d, i) => (
        <g key={d.label}>
          {d.value != null && (
            <rect x={pad.l + i * bw + bw * 0.18} width={bw * 0.64} y={y(d.value)} height={Math.max(0.5, y(0) - y(d.value))}
                  fill={d.tone ?? color}>
              <title>{d.label}: {d.value.toFixed(3)}{d.note ? ` (${d.note})` : ""}</title>
            </rect>
          )}
          <text x={pad.l + (i + 0.5) * bw} y={height - 6} textAnchor="middle">{d.label}</text>
        </g>
      ))}
      {reference != null && (
        <g>
          <line x1={pad.l} x2={w - pad.r} y1={y(reference)} y2={y(reference)} stroke="var(--oc-accent)" strokeDasharray="4 3" />
          {refLabel && <text x={w - pad.r} y={y(reference) - 4} textAnchor="end" style={{ fill: "var(--oc-accent)" }}>{refLabel}</text>}
        </g>
      )}
    </svg>
    </div>
  );
}

/** Reliability diagram: predicted vs observed rate per bin. */
export function Reliability({ bins, size = 220, ariaLabel }: {
  bins: { mean_predicted: number; observed_rate: number; n: number }[]; size?: number; ariaLabel: string;
}) {
  const pad = 26;
  const max = Math.max(0.05, ...bins.map((b) => Math.max(b.mean_predicted, b.observed_rate)));
  const s = (v: number) => pad + (v / max) * (size - 2 * pad);
  const yy = (v: number) => size - s(v);
  return (
    <svg className="chart" viewBox={`0 0 ${size} ${size}`} role="img" aria-label={ariaLabel} style={{ width: size, height: size }}>
      <line className="axis" x1={pad} y1={size - pad} x2={size - pad} y2={size - pad} />
      <line className="axis" x1={pad} y1={size - pad} x2={pad} y2={pad} />
      <line x1={s(0)} y1={yy(0)} x2={s(max)} y2={yy(max)} stroke="var(--oc-hairline-strong)" strokeDasharray="3 3" />
      <polyline fill="none" stroke="var(--oc-ev-model)" strokeWidth={1.5}
                points={bins.map((b) => `${s(b.mean_predicted)},${yy(b.observed_rate)}`).join(" ")} />
      {bins.map((b, i) => (
        <circle key={i} cx={s(b.mean_predicted)} cy={yy(b.observed_rate)} r={2.5} fill="var(--oc-ev-model)">
          <title>predicted {b.mean_predicted.toFixed(4)}, observed {b.observed_rate.toFixed(4)}, n {b.n}</title>
        </circle>
      ))}
      <text x={size / 2} y={size - 6} textAnchor="middle">predicted</text>
      <text x={8} y={size / 2} textAnchor="middle" transform={`rotate(-90 8 ${size / 2})`}>observed</text>
      <text x={size - pad} y={size - pad + 12} textAnchor="end">{max.toFixed(2)}</text>
    </svg>
  );
}
