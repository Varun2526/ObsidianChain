/**
 * Load data: upload a bulk metadata file and see what actually parsed.
 *
 * The honest bit is the outcome. This validates and correlates; it does not
 * score. The response says so and this page repeats it prominently, because
 * an upload screen that hands back a green tick invites the reader to assume
 * the alerts on the next page came from their file. They did not.
 */
import { useCallback, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { ingestFile } from "../api/client";
import type { IngestResult } from "../api/types";
import { ErrorState } from "./ErrorState";

const ACCEPTED = ".csv,.json,.xml";

export function IngestPage() {
  const input = useRef<HTMLInputElement>(null);
  const [result, setResult] = useState<IngestResult | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);

  const submit = useCallback(async (file: File) => {
    setBusy(true); setError(null); setResult(null);
    try {
      setResult(await ingestFile(file));
    } catch (cause) {
      setError(cause);
    } finally {
      setBusy(false);
    }
  }, []);

  return (
    <>
      <p><Link to="/">← Back to start</Link></p>

      <section className="panel">
        <div className="panel-head"><h2>Step 1 · Load data</h2></div>
        <div className="panel-body">
          <p className="muted small" style={{ marginTop: 0 }}>
            Upload bulk Bitcoin transaction / network metadata as{" "}
            <strong>CSV, JSON or XML</strong>. The file is parsed, normalised
            to the standard field set, validated, and correlated. Nothing is
            uploaded anywhere — the API runs locally and offline.
          </p>

          <div
            className={`dropzone${dragging ? " active" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault(); setDragging(false);
              const file = e.dataTransfer.files?.[0];
              if (file) void submit(file);
            }}
            onClick={() => input.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => { if (e.key === "Enter") input.current?.click(); }}
            aria-label="Choose a metadata file to validate"
          >
            <strong>{busy ? "Validating…" : "Drop a file here, or click to choose"}</strong>
            <span className="small faint">CSV · JSON · XML — up to 64 MB</span>
            <input
              ref={input} type="file" accept={ACCEPTED} hidden
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void submit(file);
              }}
            />
          </div>
        </div>
      </section>

      {error ? (
        <section className="panel"><ErrorState error={error} /></section>
      ) : null}

      {result ? <IngestReport result={result} /> : null}
    </>
  );
}

function IngestReport({ result }: { result: IngestResult }) {
  const v = result.validation;
  const c = result.correlation;
  return (
    <>
      <section className="panel">
        <div className="panel-head">
          <h2>Validation</h2>
          <span className={`cat cat-${v.ok ? "BLOCKCHAIN_CONTEXT" : "INSUFFICIENT_EVIDENCE"}`}>
            {v.ok ? "USABLE" : "NOT USABLE"}
          </span>
          <span className="small muted">
            {result.filename} · {v.source_format.toUpperCase()} ·{" "}
            {result.bytes.toLocaleString()} bytes
          </span>
        </div>
        <div className="panel-body">
          <div className="stat-row">
            <Stat k="Rows read" v={v.rows_read} />
            <Stat k="Rows valid" v={v.rows_valid} />
            <Stat k="Rows rejected" v={v.rows_rejected} />
            <Stat k="Transactions" v={c.transactions} />
            <Stat k="Addresses" v={c.addresses} />
            <Stat k="Source IPs" v={c.source_ips} />
            <Stat k="ASNs" v={c.asns} />
            <Stat k="Correlatable records" v={c.correlatable_records} />
          </div>

          {v.required_missing.length ? (
            <div className="banner banner-error" style={{ marginTop: 16 }}>
              <h4>Required field missing</h4>
              <p>
                {v.required_missing.join(", ")} — without it a record cannot be
                joined to the blockchain layer, so nothing was ingested.
              </p>
            </div>
          ) : null}

          {v.errors.length ? (
            <>
              <h3 className="small muted" style={{ margin: "16px 0 6px" }}>Errors</h3>
              <ul className="note" style={{ paddingLeft: 18, margin: 0 }}>
                {v.errors.map((e, i) => <li key={i}>{e}</li>)}
              </ul>
            </>
          ) : null}

          {v.warnings.length ? (
            <>
              <h3 className="small muted" style={{ margin: "16px 0 6px" }}>Warnings</h3>
              <ul className="note" style={{ paddingLeft: 18, margin: 0 }}>
                {v.warnings.map((w, i) => <li key={i}>{w}</li>)}
              </ul>
            </>
          ) : null}

          <h3 className="small muted" style={{ margin: "18px 0 6px" }}>Field coverage</h3>
          <div className="chips">
            {result.canonical_columns.map((column) => {
              const present = v.columns_present.includes(column);
              return (
                <span key={column}
                  className={`cat cat-${present ? "BLOCKCHAIN_CONTEXT" : "INSUFFICIENT_EVIDENCE"}`}
                  title={present ? "present in the upload" : "absent from the upload"}>
                  {column}
                </span>
              );
            })}
          </div>
        </div>
      </section>

      {result.preview.length ? (
        <section className="panel">
          <div className="panel-head">
            <h2>Preview</h2>
            <span className="small muted">first {result.preview.length} normalised rows</span>
          </div>
          <div className="panel-body flush" style={{ overflowX: "auto" }}>
            <table>
              <thead>
                <tr>{result.canonical_columns.map((c2) => <th key={c2}>{c2}</th>)}</tr>
              </thead>
              <tbody>
                {result.preview.map((row, i) => (
                  <tr key={i}>
                    {result.canonical_columns.map((c2) => {
                      const value = row[c2];
                      return (
                        <td key={c2} className="mono small">
                          {value === null || value === undefined
                            ? <span className="faint">n/a</span>
                            : Array.isArray(value)
                              ? value.join(", ")
                              : String(value)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      <section className="panel">
        <div className="panel-head"><h2>What happens next</h2></div>
        <div className="panel-body">
          <div className="banner banner-synthetic">
            <h4>This upload was validated, not scored</h4>
            <p>{result.next_step.why}</p>
          </div>
          <p className="muted small" style={{ marginBottom: 6 }}>
            To score a dataset, run the offline pipeline:
          </p>
          <pre className="cmd">{result.next_step.command}</pre>
          <p className="note">
            The alerts in the investigation console come from the dataset that
            pipeline last produced — not from this file.
          </p>
        </div>
      </section>
    </>
  );
}

function Stat({ k, v }: { k: string; v: number }) {
  return (
    <div className="stat">
      <span className="v">{v.toLocaleString()}</span>
      <span className="k">{k}</span>
    </div>
  );
}
