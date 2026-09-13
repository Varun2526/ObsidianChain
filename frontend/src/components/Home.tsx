/**
 * The landing experience: what this is, and what to do next.
 *
 * The investigation console is dense by design, and dropping a new user
 * straight onto a filter grid gives them no way to answer "what do I do?".
 * This page answers it in order - load, analyse, investigate - and states
 * plainly which of those steps are live right now and which are an offline
 * pipeline run. It never implies that opening the dashboard scored anything.
 */
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { fetchAlerts } from "../api/client";
import type { AlertListResponse } from "../api/types";

export function Home() {
  const [data, setData] = useState<AlertListResponse | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    fetchAlerts({ limit: 1 }, controller.signal)
      .then(setData)
      .catch((cause) => {
        if ((cause as Error)?.name !== "AbortError") setFailed(true);
      });
    return () => controller.abort();
  }, []);

  const ready = data !== null;

  return (
    <>
      <section className="panel">
        <div className="panel-body">
          <h1 style={{ margin: "0 0 6px", fontSize: 22, fontWeight: 650 }}>
            Bitcoin transaction traffic — monitoring &amp; analysis
          </h1>
          <p className="muted" style={{ margin: "0 0 4px", maxWidth: 860 }}>
            An offline forensics prototype. It correlates network-layer
            announcement observations with blockchain-layer transaction data,
            scores addresses with a calibrated model, groups them into
            entities, and ranks the result as explainable investigative leads.
          </p>
          <p className="note" style={{ maxWidth: 860 }}>
            Every network observation in this deployment is <strong>synthetic</strong>.
            It demonstrates the mechanism and validates nothing about Bitcoin.
          </p>
        </div>
      </section>

      <div className="steps">
        <StepCard
          n={1}
          title="Load data"
          done={false}
          body="Upload a bulk metadata file — CSV, JSON or XML — and see exactly what parsed, what was rejected, and what could be correlated."
          action={<Link className="cta" to="/ingest">Open the loader →</Link>}
        />
        <StepCard
          n={2}
          title="Analyse"
          done={ready}
          body={
            ready
              ? `A scored dataset is loaded: ${data!.alert_count_total.toLocaleString()} alerts from run ${data!.run_fingerprint}.`
              : failed
                ? "No scored dataset is reachable. Start the API with `make serve`."
                : "Checking for a scored dataset…"
          }
          action={
            <span className="small faint">
              Scoring is an offline pipeline run, not a button. It builds the
              as-of-t feature matrix over the whole dataset and fits on the
              temporal split.
            </span>
          }
        />
        <StepCard
          n={3}
          title="Investigate"
          done={false}
          body="Work the ranked queue. Open an alert to see why the model flagged it, the evidence behind it, who it transacted with, and where the announcements were seen."
          action={
            ready
              ? <Link className="cta" to="/alerts">Open the alert queue →</Link>
              : <span className="small faint">Available once a dataset is loaded.</span>
          }
        />
      </div>

      <section className="panel">
        <div className="panel-head"><h2>What you will see in an alert</h2></div>
        <div className="panel-body">
          <div className="explain-grid">
            <Explain title="Risk, and what it means">
              A calibrated probability from a LightGBM model, aggregated from
              member addresses to the cluster, with a severity band chosen on
              the validation split. It is a ranking signal for attention — not
              a finding and not an accusation.
            </Explain>
            <Explain title="Why it was flagged">
              The model's own SHAP contributions, in log-odds, per address.
              Model signals are kept separate from ledger facts throughout.
            </Explain>
            <Explain title="Who it dealt with">
              Co-spend membership and funding relationships, drawn as a graph.
              Only relationships the data supports are shown.
            </Explain>
            <Explain title="Where it was seen">
              Which peers announced each transaction, on what port and ASN,
              and when. A peer is a relay vantage point — never a sender, and
              never an owner.
            </Explain>
          </div>
        </div>
      </section>
    </>
  );
}

function StepCard({ n, title, body, action, done }: {
  n: number; title: string; body: string;
  action: React.ReactNode; done: boolean;
}) {
  return (
    <section className="panel step">
      <div className="panel-body">
        <div className="step-head">
          <span className={`step-n${done ? " done" : ""}`}>{done ? "✓" : n}</span>
          <h2 style={{ margin: 0, fontSize: 15 }}>{title}</h2>
        </div>
        <p className="muted small" style={{ margin: "10px 0 14px" }}>{body}</p>
        {action}
      </div>
    </section>
  );
}

function Explain({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 style={{ margin: "0 0 5px", fontSize: 13 }}>{title}</h3>
      <p className="muted small" style={{ margin: 0 }}>{children}</p>
    </div>
  );
}
