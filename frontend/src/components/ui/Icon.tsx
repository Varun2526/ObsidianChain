/**
 * A small stroke icon set, drawn inline so nothing is fetched.
 *
 * 16px grid, 1.5px stroke, round joins. Icons label nothing on their own:
 * every use sits beside a visible word or carries an aria-label.
 */
import type { SVGProps } from "react";

const PATHS = {
  overview: "M2.5 2.5h4.5v4.5h-4.5zM9 2.5h4.5v7h-4.5zM2.5 9h4.5v4.5h-4.5zM9 11.5h4.5v2h-4.5z",
  alert: "M8 2.2 14.2 13H1.8zM8 6.5v3M8 11.4v.1",
  folder: "M2 4.2c0-.7.5-1.2 1.2-1.2h3l1.4 1.6h5.2c.7 0 1.2.5 1.2 1.2v6.2c0 .7-.5 1.2-1.2 1.2H3.2c-.7 0-1.2-.5-1.2-1.2z",
  graph: "M4 4.5a1.5 1.5 0 1 0 0-.01zM12 3.5a1.5 1.5 0 1 0 0-.01zM12 12.5a1.5 1.5 0 1 0 0-.01zM5.3 4.8l5.2-1M5.2 5.5l5.6 6",
  model: "M3 13V8M6.3 13V4M9.6 13V6.5M13 13V2.5",
  flask: "M6.2 2h3.6M6.8 2v4L3 12.6c-.4.7.1 1.4.9 1.4h8.2c.8 0 1.3-.7.9-1.4L9.2 6V2M4.6 10h6.8",
  review: "M3 8.5 6.2 11.5 13 4.5",
  shield: "M8 1.8 13.2 3.8v4c0 3.2-2.2 5.4-5.2 6.4-3-1-5.2-3.2-5.2-6.4v-4z",
  users: "M6 7.2a2.2 2.2 0 1 0 0-.01zM1.8 13.5c.4-2.1 2.1-3.4 4.2-3.4s3.8 1.3 4.2 3.4M10.8 4.2a2 2 0 0 1 0 3.9M12 10.3c1.2.4 2 1.6 2.2 3.2",
  database: "M2.8 3.8c0-1 2.3-1.8 5.2-1.8s5.2.8 5.2 1.8-2.3 1.8-5.2 1.8-5.2-.8-5.2-1.8zM2.8 3.8v8.4c0 1 2.3 1.8 5.2 1.8s5.2-.8 5.2-1.8V3.8M2.8 8c0 1 2.3 1.8 5.2 1.8s5.2-.8 5.2-1.8",
  log: "M4 2.5h8M4 6h8M4 9.5h8M4 13h5",
  settings: "M8 5.6a2.4 2.4 0 1 0 0 4.8 2.4 2.4 0 0 0 0-4.8zM8 1.8v1.6M8 12.6v1.6M1.8 8h1.6M12.6 8h1.6M3.6 3.6l1.1 1.1M11.3 11.3l1.1 1.1M3.6 12.4l1.1-1.1M11.3 4.7l1.1-1.1",
  search: "M7 2.5a4.5 4.5 0 1 0 0 9 4.5 4.5 0 0 0 0-9zM10.3 10.3 13.8 13.8",
  menu: "M2.5 4h11M2.5 8h11M2.5 12h11",
  close: "M3.5 3.5l9 9M12.5 3.5l-9 9",
  arrowRight: "M2.5 8h10.5M9 4l4 4-4 4",
  arrowLeft: "M13.5 8H3M7 4 3 8l4 4",
  upstream: "M13.5 8H4M7.5 4.5 4 8l3.5 3.5M2 3v10",
  downstream: "M2.5 8H12M8.5 4.5 12 8l-3.5 3.5M14 3v10",
  both: "M1.5 8h13M4.5 5 1.5 8l3 3M11.5 5l3 3-3 3",
  plus: "M8 3v10M3 8h10",
  minus: "M3 8h10",
  fit: "M2.5 5.5v-3h3M10.5 2.5h3v3M13.5 10.5v3h-3M5.5 13.5h-3v-3",
  reset: "M3 8a5 5 0 1 0 1.5-3.6M3 2.5v2.4h2.4",
  path: "M3.5 12.5a1.5 1.5 0 1 0 0-.01zM12.5 3.5a1.5 1.5 0 1 0 0-.01zM4.6 11.4c2-2 1-4.5 3.4-5.4 2-.8 2.8-.5 3.4-1.4",
  expand: "M8 2.5v11M2.5 8h11M5 5l-2.5 3L5 11M11 5l2.5 3L11 11",
  copy: "M5.5 5.5h7v8h-7zM3.5 10.5v-8h7",
  external: "M9.5 2.5h4v4M13.5 2.5 7.5 8.5M11.5 9.5v3.5c0 .3-.2.5-.5.5h-8c-.3 0-.5-.2-.5-.5v-8c0-.3.2-.5.5-.5h3.5",
  layers: "M8 2 14 5 8 8 2 5zM2 8l6 3 6-3M2 11l6 3 6-3",
  clock: "M8 2.2a5.8 5.8 0 1 0 0 11.6A5.8 5.8 0 0 0 8 2.2zM8 4.8V8l2.2 1.4",
  list: "M5.5 4h8M5.5 8h8M5.5 12h8M2.5 4h.1M2.5 8h.1M2.5 12h.1",
  upload: "M8 10.5V2.5M4.8 5.6 8 2.5l3.2 3.1M2.5 10.5v2c0 .6.4 1 1 1h9c.6 0 1-.4 1-1v-2",
  signout: "M6 2.5H3.5c-.6 0-1 .4-1 1v9c0 .6.4 1 1 1H6M10 5l3 3-3 3M13 8H6",
  node: "M8 4.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7z",
  tx: "M4 4h8v8H4z",
  filter: "M2.5 3h11l-4.3 5.2v4.3l-2.4 1V8.2z",
  info: "M8 2.2a5.8 5.8 0 1 0 0 11.6A5.8 5.8 0 0 0 8 2.2zM8 7.2v4M8 5v.1",
} as const;

export type IconName = keyof typeof PATHS;

export function Icon({ name, size = 16, ...rest }: { name: IconName; size?: number } & SVGProps<SVGSVGElement>) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      <path d={PATHS[name]} />
    </svg>
  );
}
