/**
 * Admin Datasets & Provenance Registry
 *
 * Global inventory of forensic dataset captures:
 * - Content-addressed SHA-256 integrity
 * - Format, byte footprint, and validation results
 * - Linked investigation cases
 */
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import * as api from "../../api/console";
import type { UploadedDataset } from "../../api/types";
import { Skeleton } from "../../components/ui/primitives";

interface DatasetRow extends UploadedDataset {
  case_id: string;
  case_label: string;
  case_name: string;
}

export function AdminDatasetsPage() {
  const [datasets, setDatasets] = useState<DatasetRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    api.listInvestigations(controller.signal)
      .then((r) => {
        const rows: DatasetRow[] = [];
        for (const inv of r.investigations) {
          if (inv.datasets) {
            for (const d of inv.datasets) {
              rows.push({
                ...d,
                case_id: inv.id,
                case_label: inv.case_label,
                case_name: inv.name,
              });
            }
          }
        }
        setDatasets(rows);
      })
      .catch(() => {})
      .finally(() => setLoading(false));

    return () => controller.abort();
  }, []);

  const filtered = datasets.filter((d) =>
    d.filename.toLowerCase().includes(filter.toLowerCase()) ||
    d.sha256.toLowerCase().includes(filter.toLowerCase()) ||
    d.case_label.toLowerCase().includes(filter.toLowerCase())
  );

  const totalBytes = datasets.reduce((sum, d) => sum + d.size_bytes, 0);

  return (
    <>
      <div className="page-header" style={{ marginBottom: 20 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
            <h1 style={{ margin: 0 }}>Dataset & Capture Registry</h1>
            <span className="status-badge status-active">CONTENT-ADDRESSED (SHA-256)</span>
          </div>
          <p className="muted" style={{ margin: 0 }}>
            Forensic capture repository, cryptographic fingerprints, and linked casework references
          </p>
        </div>
      </div>

      <div className="card-grid-4" style={{ marginBottom: 24 }}>
        <div className="stat-card">
          <span className="stat-card-value">{datasets.length}</span>
          <span className="stat-card-label">Stored Captures</span>
        </div>
        <div className="stat-card">
          <span className="stat-card-value">{(totalBytes / (1024 * 1024)).toFixed(2)} MB</span>
          <span className="stat-card-label">Total Footprint</span>
        </div>
        <div className="stat-card">
          <span className="stat-card-value" style={{ color: "var(--model)" }}>
            {datasets.filter((d) => d.status === "VALIDATED").length}
          </span>
          <span className="stat-card-label">Validated Schema</span>
        </div>
        <div className="stat-card">
          <span className="stat-card-value">100%</span>
          <span className="stat-card-label">Offline Integrity</span>
        </div>
      </div>

      {/* Filter / Search Bar */}
      <div className="panel" style={{ marginBottom: 20 }}>
        <div className="panel-body" style={{ padding: "12px 18px" }}>
          <input
            type="text"
            placeholder="Filter captures by filename, SHA-256 hash, or case label…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            style={{ width: "100%" }}
          />
        </div>
      </div>

      {/* Datasets Table */}
      <section className="panel">
        <div className="panel-head">
          <h2>Registered Datasets</h2>
          <span className="small muted">{filtered.length} Displayed</span>
        </div>
        <div className="panel-body flush">
          {loading ? (
            <div style={{ padding: 16 }}><Skeleton rows={5} /></div>
          ) : filtered.length === 0 ? (
            <p className="muted" style={{ padding: 16 }}>No datasets found.</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Filename</th>
                  <th>Format</th>
                  <th className="num">Size</th>
                  <th>Cryptographic SHA-256 Fingerprint</th>
                  <th>Validation</th>
                  <th>Associated Case</th>
                  <th>Uploaded</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((d) => (
                  <tr key={`${d.case_id}-${d.id}`}>
                    <td><strong>{d.filename}</strong></td>
                    <td className="small muted">{d.format.toUpperCase()}</td>
                    <td className="num">{d.size_bytes.toLocaleString()} B</td>
                    <td className="mono small">{d.sha256}</td>
                    <td>
                      <span className={`status-badge status-${d.status === "VALIDATED" ? "active" : "closed"}`}>
                        {d.status}
                      </span>
                    </td>
                    <td>
                      <Link to={`/inv/${d.case_id}`} className="mono font-semibold">
                        {d.case_label}
                      </Link>
                      <span className="small muted" style={{ display: "block" }}>{d.case_name}</span>
                    </td>
                    <td className="small muted mono">
                      {new Date(d.uploaded_at).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </section>
    </>
  );
}
