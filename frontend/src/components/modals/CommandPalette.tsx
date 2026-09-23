/**
 * Search everything, from anywhere (Cmd/Ctrl-K or "/").
 *
 * Results come from two real sources and nowhere else: the backend search
 * (alert ids, transaction ids, address prefixes over the chain index) and
 * the investigations this user may read. Nothing is suggested that the API
 * did not return.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import * as api from "../../api/console";
import { search } from "../../api/intel";
import type { SearchResult } from "../../api/intel";
import type { Investigation } from "../../api/types";
import { Icon } from "../ui/Icon";

interface Item { key: string; kind: string; label: string; detail?: string; to: string }

function toItem(r: SearchResult): Item {
  if (r.kind === "alert") return { key: `a${r.id}`, kind: "Alert", label: `Cluster ${r.id.split(":")[1]}`, detail: r.id, to: `/alerts/${encodeURIComponent(r.id)}` };
  if (r.kind === "transaction") return { key: `t${r.id}`, kind: "Transaction", label: r.id,
    detail: r.detail?.timestep != null ? `timestep ${r.detail.timestep} · ${r.detail.n_inputs} in · ${r.detail.n_outputs} out` : undefined, to: `/tx/${r.id}` };
  return { key: `d${r.id}`, kind: "Address", label: r.id, to: `/entity/${encodeURIComponent(r.id)}` };
}

export function CommandPalette({ open, onClose }: { open: boolean; onClose: () => void }) {
  const navigate = useNavigate();
  const input = useRef<HTMLInputElement>(null);
  const [q, setQ] = useState("");
  const [remote, setRemote] = useState<SearchResult[]>([]);
  const [cases, setCases] = useState<Investigation[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState(0);

  useEffect(() => {
    if (!open) { setQ(""); setRemote([]); setActive(0); setError(null); return; }
    input.current?.focus();
    api.listInvestigations().then((r) => setCases(r.investigations)).catch(() => setCases([]));
  }, [open]);

  useEffect(() => {
    const term = q.trim();
    if (!open || term.length < 2) { setRemote([]); return; }
    const ctrl = new AbortController();
    const t = window.setTimeout(() => {
      setBusy(true);
      search(term, ctrl.signal)
        .then((r) => { setRemote(r.results); setError(null); setBusy(false); })
        .catch((e: unknown) => {
          if ((e as Error)?.name === "AbortError") return;
          setRemote([]); setBusy(false);
          setError((e as Error)?.message ?? "Search failed");
        });
    }, 160);
    return () => { window.clearTimeout(t); ctrl.abort(); };
  }, [q, open]);

  const items = useMemo<Item[]>(() => {
    const term = q.trim().toLowerCase();
    const caseItems: Item[] = term
      ? cases.filter((c) => `${c.case_label} ${c.name}`.toLowerCase().includes(term)).slice(0, 6)
          .map((c) => ({ key: `c${c.id}`, kind: "Investigation", label: `${c.case_label} · ${c.name}`, detail: c.status, to: `/inv/${c.id}` }))
      : [];
    return [...remote.map(toItem), ...caseItems];
  }, [remote, cases, q]);

  useEffect(() => { setActive(0); }, [items.length]);

  if (!open) return null;

  const go = (item: Item | undefined) => {
    if (!item) return;
    onClose();
    navigate(item.to);
  };

  return (
    <div className="modal-overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="palette" role="dialog" aria-modal="true" aria-label="Search">
        <div className="palette-input">
          <Icon name="search" />
          <input
            ref={input}
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Address, transaction id, alert id or investigation"
            aria-label="Search"
            role="combobox"
            aria-expanded={items.length > 0}
            aria-controls="palette-results"
            aria-activedescendant={items[active] ? `pal-${items[active]!.key}` : undefined}
            onKeyDown={(e) => {
              if (e.key === "Escape") { e.preventDefault(); onClose(); }
              else if (e.key === "ArrowDown") { e.preventDefault(); setActive((a) => Math.min(items.length - 1, a + 1)); }
              else if (e.key === "ArrowUp") { e.preventDefault(); setActive((a) => Math.max(0, a - 1)); }
              else if (e.key === "Enter") { e.preventDefault(); go(items[active]); }
            }}
          />
          {busy && <span className="small faint">Searching</span>}
        </div>
        <div className="palette-body" id="palette-results" role="listbox" aria-label="Results">
          {q.trim().length < 2 && (
            <div className="palette-empty">
              Type at least two characters. Addresses match by prefix (four or more characters), transactions and alert ids exactly.
            </div>
          )}
          {q.trim().length >= 2 && !busy && items.length === 0 && (
            <div className="palette-empty">{error ?? "No address, transaction, alert or investigation matches."}</div>
          )}
          {items.map((it, i) => (
            <button key={it.key} id={`pal-${it.key}`} type="button" role="option" aria-selected={i === active}
                    className="palette-item" onMouseEnter={() => setActive(i)} onClick={() => go(it)}>
              <span className="kind">{it.kind}</span>
              <span className={`label ${it.kind === "Address" || it.kind === "Transaction" ? "mono" : ""}`}>{it.label}</span>
              {it.detail && <span className="small faint truncate" style={{ maxWidth: 220 }}>{it.detail}</span>}
            </button>
          ))}
        </div>
        <div className="palette-foot"><span><kbd>↑</kbd> <kbd>↓</kbd> move</span><span><kbd>Enter</kbd> open</span><span><kbd>Esc</kbd> close</span></div>
      </div>
    </div>
  );
}
