import pandas as pd, numpy as np
wc=pd.read_csv('data/raw/wallets_classes.csv'); print('wallet classes',wc['class'].value_counts().to_dict(), 'dup addr',wc.address.duplicated().sum())
tc=pd.read_csv('data/raw/txs_classes.csv'); print('tx classes',tc['class'].value_counts().to_dict())
wf=pd.read_csv('data/raw/wallets_features.csv',usecols=['address','Time step'])
s=wf.groupby('address')['Time step'].agg(['min','max','nunique'])
s=s.join(wc.set_index('address')['class'])
print('addresses',len(s)); print('multi-step addresses',(s['nunique']>1).mean())
sp=((s['min']<=34)&(s['max']>34))|((s['min']<=41)&(s['max']>41))
lab=s['class'].isin([1,2])
print('labeled addresses',lab.sum(),'spanners among labeled',(sp&lab).sum(), 'frac',(sp&lab).sum()/lab.sum())
print('illicit rate spanners',(s.loc[sp&lab,'class']==1).mean(),' non-spanners',(s.loc[~sp&lab,'class']==1).mean())
print('multi-step labeled illicit rate',(s.loc[lab&(s['nunique']>1),'class']==1).mean(),'single',(s.loc[lab&(s['nunique']==1),'class']==1).mean())
# unknown by step
first=s['min']; print('unknown frac overall',(s['class']==3).mean())
