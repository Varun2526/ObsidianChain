import pandas as pd, numpy as np, joblib, json
from sklearn.metrics import average_precision_score, roc_auc_score
U="/mnt/user-data/uploads/obsidianchain/data/models/"; D=U+"ps_native/datasets/"; V=U+"ps_native/v1/"
tr=pd.read_parquet(D+"train.parquet"); va=pd.read_parquet(D+"validation.parquet"); te=pd.read_parquet(D+"test.parquet")
art=joblib.load(V+"model.joblib"); mdl=art["model"]; cal=art["calibrator"]; FE=list(art["features"]); BANDS=art["severity_bands"]
calj={"severity_bands":[{"band":"CRITICAL","support":840,"validation_precision":0.9},{"band":"HIGH","support":1452,"validation_precision":0.75},{"band":"MEDIUM","support":2602,"validation_precision":0.5}]}
def raw(d):
    X=np.nan_to_num(d[FE].to_numpy(dtype=np.float32),nan=0.0,posinf=0.0,neginf=0.0)
    return mdl.predict_proba(X)[:,1]

print("="*78); print("A. WERE THE BAND THRESHOLDS FIT ON RAW OR CALIBRATED SCORES?")
print("calibration.json claims (validation):", [(b['band'],b['support'],b['validation_precision']) for b in calj['severity_bands']])
yv=va.y.to_numpy(); rv=raw(va); cv=cal.predict(rv)
for nm,s in [("raw",rv),("calibrated",cv)]:
    print(f"\n  applying thresholds to {nm} validation scores:")
    for b in BANDS:
        m=s>=b["threshold"]
        print(f"    {b['band']:9s} thr>={b['threshold']:.4f}  n_at_or_above={m.sum():6,d} precision={yv[m].mean() if m.sum() else float('nan'):.4f}"
              f"   | claimed support={b.get('support')} claimed_prec={b.get('validation_precision')}")

print("\n"+"="*78); print("B. PREVALENCE-CONTROLLED PERFORMANCE (normalised AP = (AP-prev)/(1-prev))")
for nm,d in [("validation",va),("test",te)]:
    y=d.y.to_numpy(); c=cal.predict(raw(d)); ap=average_precision_score(y,c); p=y.mean()
    print(f"  {nm:11s} prev={p:.4f} AP={ap:.4f} nAP={(ap-p)/(1-p):.4f} lift={ap/p:.2f}x ROC={roc_auc_score(y,c):.4f}")

print("\n"+"="*78); print("C. SPLIT DESIGN CONSEQUENCE: lifespan/activity distribution across splits")
print("(build_ps_dataset drops boundary-spanning addresses, so each split keeps only")
print(" addresses whose ENTIRE life fits inside it -> train allows 34 steps, test only 8)")
key=["n_txs_asof_t","active_duration_seconds","n_sent_asof_t","n_recv_asof_t","btc_recv_total_asof_t",
     "cluster_size_asof_t","unique_counterparties_asof_t","input_count","output_count","tx_velocity_per_hour"]
rows=[]
for f in key:
    r={"feature":f}
    for nm,d in [("train",tr),("validation",va),("test",te)]:
        r[f"{nm}_mean"]=d[f].mean(); r[f"{nm}_pct_zero"]=100*(d[f]==0).mean()
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False,float_format=lambda x:f"{x:.3f}"))

print("\n"+"="*78); print("D. MULTI-TX ADDRESSES: share with any history at all")
for nm,d in [("train",tr),("validation",va),("test",te)]:
    print(f"  {nm:11s} share with n_txs_asof_t>0 = {(d.n_txs_asof_t>0).mean():.4f}   "
          f"max_n_txs={d.n_txs_asof_t.max():.0f}  mean_active_days={(d.active_duration_seconds/86400).mean():.2f}")

print("\n"+"="*78); print("E. IS THE TEST DROP EXPLAINED BY SHORT-LIVED ADDRESSES? stratify test by history")
y=te.y.to_numpy(); c=cal.predict(raw(te))
for lab,m in [("no history (n_txs==0)",te.n_txs_asof_t.to_numpy()==0),
              ("has history (n_txs>0)",te.n_txs_asof_t.to_numpy()>0)]:
    if m.sum()>10 and 0<y[m].sum()<m.sum():
        ap=average_precision_score(y[m],c[m]); p=y[m].mean()
        print(f"  {lab:24s} n={m.sum():6,d} prev={p:.4f} AP={ap:.4f} nAP={(ap-p)/(1-p):.4f} ROC={roc_auc_score(y[m],c[m]):.4f}")
# same for validation to compare like with like
yv2=va.y.to_numpy(); cv2=cal.predict(raw(va))
print("  -- validation for comparison --")
for lab,m in [("no history (n_txs==0)",va.n_txs_asof_t.to_numpy()==0),
              ("has history (n_txs>0)",va.n_txs_asof_t.to_numpy()>0)]:
    if m.sum()>10 and 0<yv2[m].sum()<m.sum():
        ap=average_precision_score(yv2[m],cv2[m]); p=yv2[m].mean()
        print(f"  {lab:24s} n={m.sum():6,d} prev={p:.4f} AP={ap:.4f} nAP={(ap-p)/(1-p):.4f} ROC={roc_auc_score(yv2[m],cv2[m]):.4f}")
