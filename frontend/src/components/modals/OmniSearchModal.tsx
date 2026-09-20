/**
 * Global Omni-Search Modal (⌘K).
 *
 * Inspired by Palantir Object Explorer: allows instant cross-entity discovery
 * across Cases, Alerts, Clusters, Addresses, Transactions, and Network Observations.
 */
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import * as api from "../../api/console";
import { fetchAlerts } from "../../api/client";
import type { AlertSummary, Investigation } from "../../api/types";

interface SearchResultItem {
  id: string;
  type: "CASE" | "ALERT" | "CLUSTER" | "ADDRESS" | "TRANSACTION" | "NETWORK";
  title: string;
  subtitle: string;
  badge?: string;
  path: string;
}

export function OmniSearchModal({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState("");
  const [cases, setCases] = useState<Investigation[]>([]);
  const [alerts, setAlerts] = useState<AlertSummary[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);

  useEffect(() => {
    if (open) {
      inputRef.current?.focus();
      // Load index sources
      api.listInvestigations().then((r) => setCases(r.investigations)).catch(() => {});
      fetchAlerts({ limit: 50 }).then((r) => setAlerts(r.alerts)).catch(() => {});
    } else {
      setQuery("");
      setSelectedIndex(0);
    }
  }, [open]);

  // Close on Escape or click outside
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && open) {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open, onClose]);

  // Match entities
  const q = query.trim().toLowerCase();
  const results: SearchResultItem[] = [];

  if (q) {
    // 1. Cases
    cases.forEach((c) => {
      if (
        c.case_label.toLowerCase().includes(q) ||
        c.name.toLowerCase().includes(q) ||
        (c.description && c.description.toLowerCase().includes(q))
      ) {
        results.push({
          id: `case-${c.id}`,
          type: "CASE",
          title: `${c.case_label}: ${c.name}`,
          subtitle: `Status: ${c.status} · Updated ${new Date(c.updated_at).toLocaleDateString()}`,
          badge: c.status,
          path: `/inv/${c.id}`,
        });
      }
    });

    // 2. Alerts & Clusters
    alerts.forEach((a) => {
      const clusterStr = `cluster #${a.cluster_id}`.toLowerCase();
      const alertStr = a.alert_id.toLowerCase();
      if (clusterStr.includes(q) || alertStr.includes(q) || (a.risk_score && String(Math.round(a.risk_score * 100)).includes(q))) {
        results.push({
          id: `alert-${a.alert_id}`,
          type: "CLUSTER",
          title: `Cluster #${a.cluster_id}`,
          subtitle: `Alert ID: ${a.alert_id} · Risk: ${a.risk_score ? `${Math.round(a.risk_score * 100)}%` : "N/A"}`,
          badge: a.severity,
          path: `/alerts/${a.alert_id}`,
        });
      }
    });

    // 3. Addresses or TXIDs detected by format
    if (q.startsWith("1") || q.startsWith("3") || q.startsWith("bc1") || q.length >= 8) {
      results.push({
        id: `addr-${q}`,
        type: "ADDRESS",
        title: `Address: ${query.trim()}`,
        subtitle: `Inspect transaction history, counterparty degrees & cluster co-spend`,
        badge: "FORENSIC PROFILE",
        path: `/alerts?search=${encodeURIComponent(query.trim())}`,
      });
    }

    if (q.includes("tx") || q.length === 64) {
      results.push({
        id: `tx-${q}`,
        type: "TRANSACTION",
        title: `Transaction: ${query.trim()}`,
        subtitle: `Inspect input/output amounts, fee ratio, and script type`,
        badge: "ON-CHAIN TX",
        path: `/alerts?search=${encodeURIComponent(query.trim())}`,
      });
    }

    // 4. IP / Network Observations
    if (/^\d{1,3}\.\d{1,3}/.test(q)) {
      results.push({
        id: `ip-${q}`,
        type: "NETWORK",
        title: `Peer IP: ${query.trim()}`,
        subtitle: `Correlate P2P propagation timeline, peer announcements & ASN telemetry`,
        badge: "NETWORK EVIDENCE",
        path: `/alerts?search=${encodeURIComponent(query.trim())}`,
      });
    }
  }

  // Keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((i) => Math.min(results.length - 1, i + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((i) => Math.max(0, i - 1));
    } else if (e.key === "Enter" && results[selectedIndex]) {
      e.preventDefault();
      navigate(results[selectedIndex].path);
      onClose();
    }
  };

  if (!open) return null;

  return (
    <div className="omni-search-backdrop" onClick={onClose}>
      <div
        className="omni-search-container"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        <div className="omni-search-header">
          <svg className="omni-search-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            ref={inputRef}
            type="text"
            className="omni-search-input"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelectedIndex(0);
            }}
            placeholder="Search cases, TXIDs, addresses, clusters, peer IPs… (Esc to close)"
          />
          <kbd className="omni-search-esc">ESC</kbd>
        </div>

        <div className="omni-search-body">
          {!q ? (
            <div className="omni-search-hint">
              <span className="hint-title">Universal Investigative Search</span>
            </div>
          ) : results.length === 0 ? (
            <div className="omni-search-empty">
              <p>No matching entities found for <strong>"{query}"</strong></p>
            </div>
          ) : (
            <div className="omni-search-results">
              {results.map((item, idx) => (
                <div
                  key={item.id}
                  className={`omni-search-item${selectedIndex === idx ? " selected" : ""}`}
                  onClick={() => {
                    navigate(item.path);
                    onClose();
                  }}
                  onMouseEnter={() => setSelectedIndex(idx)}
                >
                  <div className="omni-item-type-badge">{item.type}</div>
                  <div className="omni-item-content">
                    <div className="omni-item-title-row">
                      <span className="omni-item-title">{item.title}</span>
                      {item.badge && <span className="omni-item-tag">{item.badge}</span>}
                    </div>
                    <span className="omni-item-subtitle">{item.subtitle}</span>
                  </div>
                  <span className="omni-item-jump">↵</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
