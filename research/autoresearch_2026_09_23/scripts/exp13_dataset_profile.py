import pandas as pd, numpy as np
P='data/models/ps_native/datasets/'
tr=pd.read_parquet(P+'train.parquet'); va=pd.read_parquet(P+'validation.parquet')
tr['split']='train'; va['split']='val'; d=pd.concat([tr,va],ignore_index=True)
d['step']=((d.timestamp-1400000000)/1209600+1).round().astype(int)
F=[c for c in tr.columns if c not in('address','txid','timestamp','y','split')]
print('== prevalence'); print(d.groupby('split').y.agg(['size','sum','mean']))
g=d.groupby('step').y.agg(['size','sum','mean']); print(g.to_string())
print('== distinct values / zero frac / missing / inf')
for c in F:
    x=d[c]; print(f'{c:30s} nuniq={x.nunique():7d} zero={np.mean(x==0):.3f} nan={x.isna().mean():.3f} inf={np.isinf(x.astype(float)).sum()} neg={np.mean(x<0):.3f} p50={x.median():.4g} p99={x.quantile(.99):.4g} max={x.max():.4g}')
print('== cold start frac (n_txs_asof_t==0)', (d.n_txs_asof_t==0).mean(), 'by y', d.groupby('y').apply(lambda s:(s.n_txs_asof_t==0).mean()).to_dict())
# history time features quantized?
for c in ['active_duration_seconds','gap_since_last_tx']:
    nz=d.loc[d[c]>0,c]; print(c,'nonzero',len(nz),'frac multiple of 1209600',np.mean(np.isclose(nz%1209600,0)))
print('== exact duplicate feature vectors')
dup=d.duplicated(F,keep=False); print('rows in dup groups',dup.mean())
grp=d[dup].groupby(F).y.agg(['size','mean'])
conf=grp[(grp['mean']>0)&(grp['mean']<1)]
print('dup groups',len(grp),'conflicting groups',len(conf),'rows in conflict',conf['size'].sum(), 'positives in conflicting', (conf['size']*conf['mean']).sum())
print('== txid shared across addresses (same tx features)')
t=d.groupby('txid').agg(n=('address','size'),ym=('y','mean'),split=('split','first'))
print('addresses per txid: dist',t.n.describe().to_dict()); print('rows whose txid shared',(d.txid.map(t.n)>1).mean())
print('txids w/ mixed labels',((t.ym>0)&(t.ym<1)).sum(),'of multi',(t.n>1).sum())
pos=d[d.y==1]; print('positives whose txid has >1 addr',(pos.txid.map(t.n)>1).mean(),' top txid pos count', pos.txid.value_counts().head(5).to_dict())
print('frac positives in top 10 txids', pos.txid.value_counts().head(10).sum()/len(pos))
print('== univariate AUC per feature (val and train)')
from sklearn.metrics import roc_auc_score, average_precision_score
for c in F:
    a=roc_auc_score(tr.y,tr[c]); b=roc_auc_score(va.y,va[c]); print(f'{c:30s} trAUC={a:.3f} vaAUC={b:.3f}')
print('== correlations > .95')
C=d[F].corr(method='spearman').abs(); 
for i,a in enumerate(F):
    for b in F[i+1:]:
        if C.loc[a,b]>.95: print(a,b,round(C.loc[a,b],3))
