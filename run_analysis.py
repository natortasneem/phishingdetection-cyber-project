from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance
warnings.filterwarnings('ignore')
RANDOM_STATE=42
BASE=Path(__file__).resolve().parent; DATA=BASE/'data'; FIG=BASE/'figures'; OUT=BASE/'outputs'
for p in [DATA,FIG,OUT]: p.mkdir(exist_ok=True)
df=pd.read_csv(DATA/'5.urldata.csv').rename(columns=lambda c:c.replace('/','_').replace(' ','_'))
orig=[c for c in df.columns if c not in ['Domain','Label']]
summary={'shape':list(df.shape),'columns':list(df.columns),'missing_values_total':int(df.isna().sum().sum()),'class_counts':{str(k):int(v) for k,v in df['Label'].value_counts().sort_index().items()},'class_percent':{str(k):float(v) for k,v in (df['Label'].value_counts(normalize=True).sort_index()*100).round(2).items()},'single_value_columns':[c for c in df.columns if df[c].nunique(dropna=False)<=1],'duplicated_rows':int(df.duplicated().sum()),'duplicated_domains':int(df['Domain'].duplicated().sum()),'unique_domains':int(df['Domain'].nunique()),'has_temporal_column':bool(any(('date' in c.lower() or 'time' in c.lower() or 'timestamp' in c.lower()) for c in df.columns))}
pd.DataFrame({'dtype':df.dtypes.astype(str),'missing_count':df.isna().sum(),'unique_values':df.nunique(dropna=False),'example_value':df.iloc[0].astype(str)}).to_csv(OUT/'data_inspection.csv')
# Figures
plt.figure(figsize=(5,4)); df['Label'].value_counts().sort_index().plot(kind='bar'); plt.xticks([0,1],['Legitimate (0)','Phishing (1)'],rotation=0); plt.ylabel('Count'); plt.title('Class Distribution'); plt.tight_layout(); plt.savefig(FIG/'class_distribution.png',dpi=180); plt.close()
plt.figure(figsize=(7,4))
for lab,name in [(0,'Legitimate'),(1,'Phishing')]: df.loc[df['Label']==lab,'URL_Depth'].plot(kind='hist',bins=18,alpha=.55,label=name)
plt.title('URL Depth Distribution by Class'); plt.xlabel('URL Depth'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'url_depth_distribution.png',dpi=180); plt.close()
binary=[c for c in orig if df[c].nunique()==2]; mean=df.groupby('Label')[binary].mean().T; mean['abs_gap']=(mean[1]-mean[0]).abs(); topmean=mean.sort_values('abs_gap',ascending=False).head(12); topmean.to_csv(OUT/'binary_prevalence_by_class.csv'); topmean[[0,1]].plot(kind='barh',figsize=(8,6)); plt.title('Top Binary Feature Prevalence by Class'); plt.xlabel('Feature mean / prevalence'); plt.tight_layout(); plt.savefig(FIG/'binary_feature_prevalence.png',dpi=180); plt.close()
for col in ['Web_Traffic','Domain_Age','DNS_Record','Prefix_Suffix','TinyURL']:
    pd.crosstab(df[col],df['Label'],normalize='index').rename(columns={0:'Legitimate_rate',1:'Phishing_rate'}).round(4).to_csv(OUT/f'crosstab_{col}.csv')
df.groupby('Label')[['URL_Depth','URL_Length','Redirection','Web_Forwards']].agg(['mean','median','std']).round(4).to_csv(OUT/'group_summary.csv')
# Correlation
spearman=df[orig+['Label']].corr(method='spearman'); topcorr=spearman['Label'].drop('Label').abs().sort_values(ascending=False).head(12); topcorr.to_csv(OUT/'top_spearman_correlations.csv'); topcorr.sort_values().plot(kind='barh',figsize=(8,5)); plt.title('Top Absolute Spearman Correlations with Label'); plt.xlabel('|Spearman rho|'); plt.tight_layout(); plt.savefig(FIG/'top_spearman_correlations.png',dpi=180); plt.close(); summary['top_spearman_abs']={k:float(v) for k,v in topcorr.round(4).items()}
pairs=[]; ca=spearman.drop(index='Label',columns='Label').abs(); cols=list(ca.columns)
for i in range(len(cols)):
    for j in range(i+1,len(cols)):
        val=float(ca.iloc[i,j])
        if val>=.7: pairs.append({'feature_1':cols[i],'feature_2':cols[j],'abs_spearman':val})
redund=pd.DataFrame(pairs).sort_values('abs_spearman',ascending=False) if pairs else pd.DataFrame(columns=['feature_1','feature_2','abs_spearman']); redund.to_csv(OUT/'high_correlation_pairs.csv',index=False); summary['high_corr_pairs_ge_0_7']=redund.head(10).to_dict(orient='records')
# Features
def domfeat(domain):
    s=str(domain).strip().lower(); parts=s.split('.'); tld=parts[-1] if len(parts)>1 else ''; common={'com','org','net','edu','gov','co','io','jp','uk','de','fr','ru','br','in','au','ca'}
    return pd.Series({'domain_length':len(s),'domain_digit_count':sum(ch.isdigit() for ch in s),'domain_hyphen_count':s.count('-'),'domain_dot_count':s.count('.'),'subdomain_count':max(len(parts)-2,0),'has_www':int(s.startswith('www.')),'tld_length':len(tld),'is_common_tld':int(tld in common)})
eng=df['Domain'].apply(domfeat); df_fe=pd.concat([df,eng],axis=1); engcols=list(eng.columns); feats=orig+engcols; df_fe[['Domain']+feats+['Label']].to_csv(DATA/'phishing_processed_features.csv',index=False); summary['engineered_features']=engcols
X=df_fe[feats]; y=df_fe['Label']; groups=df_fe['Domain']; Xtr,Xte,ytr,yte,gtr,gte=train_test_split(X,y,groups,test_size=.2,stratify=y,random_state=RANDOM_STATE); gss=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=RANDOM_STATE); tri,tei=next(gss.split(X,y,groups=groups)); Xgtr,Xgte=X.iloc[tri],X.iloc[tei]; ygtr,ygte=y.iloc[tri],y.iloc[tei]
def models():
    return {'Logistic Regression':Pipeline([('scaler',StandardScaler()),('model',LogisticRegression(max_iter=2000,class_weight='balanced',random_state=RANDOM_STATE))]),'Decision Tree':DecisionTreeClassifier(max_depth=8,min_samples_leaf=20,random_state=RANDOM_STATE,class_weight='balanced'),'Random Forest':RandomForestClassifier(n_estimators=120,min_samples_leaf=2,random_state=RANDOM_STATE,class_weight='balanced',n_jobs=1),'Gradient Boosting':GradientBoostingClassifier(random_state=RANDOM_STATE)}
def ev(model,Xa,Xb,ya,yb):
    model.fit(Xa,ya); pred=model.predict(Xb); score=model.predict_proba(Xb)[:,1] if hasattr(model,'predict_proba') else pred; tn,fp,fn,tp=confusion_matrix(yb,pred).ravel(); return {'Accuracy':accuracy_score(yb,pred),'Precision':precision_score(yb,pred,zero_division=0),'Recall':recall_score(yb,pred,zero_division=0),'F1':f1_score(yb,pred,zero_division=0),'F2':fbeta_score(yb,pred,beta=2,zero_division=0),'MCC':matthews_corrcoef(yb,pred),'ROC_AUC':roc_auc_score(yb,score),'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)},model,pred,score
rr=[]; gr=[]; fitted={}; pc={}
for n,m in models().items():
    r,fit,p,s=ev(m,Xtr,Xte,ytr,yte); r['Model']=n; rr.append(r); fitted[n]=fit; pc[n]=(p,s)
    rg,_,_,_=ev(m,Xgtr,Xgte,ygtr,ygte); rg['Model']=n; gr.append(rg)
random_df=pd.DataFrame(rr).set_index('Model').sort_values('F1',ascending=False); group_df=pd.DataFrame(gr).set_index('Model').sort_values('F1',ascending=False); random_df.to_csv(OUT/'model_results_random_split.csv'); group_df.to_csv(OUT/'model_results_group_split.csv')
# 3-fold CV to keep runtime reliable.
cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=RANDOM_STATE); scoring={'accuracy':'accuracy','precision':'precision','recall':'recall','f1':'f1','roc_auc':'roc_auc'}; cvrows=[]
for n,m in models().items():
    sc=cross_validate(m,X,y,cv=cv,scoring=scoring,n_jobs=1); row={'Model':n}
    for met in scoring: vals=sc[f'test_{met}']; row[f'{met}_mean']=float(np.mean(vals)); row[f'{met}_std']=float(np.std(vals))
    cvrows.append(row)
cv_df=pd.DataFrame(cvrows).set_index('Model').sort_values('f1_mean',ascending=False); cv_df.to_csv(OUT/'cross_validation_results.csv')
# plots and errors
random_df[['Accuracy','Precision','Recall','F1','ROC_AUC']].sort_values('F1').plot(kind='barh',figsize=(8,5)); plt.title('Model Comparison - Random Stratified Test Split'); plt.xlabel('Metric value'); plt.xlim(0,1); plt.tight_layout(); plt.savefig(FIG/'model_comparison.png',dpi=180); plt.close()
best=random_df['F1'].idxmax(); best_model=fitted[best]; best_pred,best_score=pc[best]; cm=confusion_matrix(yte,best_pred); plt.figure(figsize=(5,4)); plt.imshow(cm); plt.title(f'Confusion Matrix - {best}'); plt.xlabel('Predicted label'); plt.ylabel('True label'); plt.xticks([0,1],['Legitimate','Phishing']); plt.yticks([0,1],['Legitimate','Phishing']);
for (i,j),v in np.ndenumerate(cm): plt.text(j,i,str(v),ha='center',va='center')
plt.tight_layout(); plt.savefig(FIG/'confusion_matrix_best.png',dpi=180); plt.close()
if hasattr(best_model,'feature_importances_'): imps=pd.Series(best_model.feature_importances_,index=feats).sort_values(ascending=False)
else: imps=pd.Series(permutation_importance(best_model,Xte,yte,n_repeats=3,random_state=RANDOM_STATE,n_jobs=1).importances_mean,index=feats).sort_values(ascending=False)
imps.to_csv(OUT/'feature_importance_best_model.csv'); imps.head(15).sort_values().plot(kind='barh',figsize=(8,5)); plt.title(f'Top Feature Importances - {best}'); plt.xlabel('Importance'); plt.tight_layout(); plt.savefig(FIG/'feature_importance.png',dpi=180); plt.close()
err=df_fe.loc[Xte.index,['Domain','Label']+feats].copy(); err['predicted']=best_pred; err['phishing_score']=best_score; eo=err[err['Label']!=err['predicted']]; eo[(eo['Label']==1)&(eo['predicted']==0)].sort_values('phishing_score').head(10).to_csv(OUT/'false_negatives_examples.csv',index=False); eo[(eo['Label']==0)&(eo['predicted']==1)].sort_values('phishing_score',ascending=False).head(10).to_csv(OUT/'false_positives_examples.csv',index=False)
th=[]
for t in [.30,.35,.40,.45,.50,.60,.70]:
    pred=(best_score>=t).astype(int); tn,fp,fn,tp=confusion_matrix(yte,pred).ravel(); th.append({'threshold':t,'precision':precision_score(yte,pred,zero_division=0),'recall':recall_score(yte,pred,zero_division=0),'f1':f1_score(yte,pred,zero_division=0),'f2':fbeta_score(yte,pred,beta=2,zero_division=0),'FP':int(fp),'FN':int(fn),'TP':int(tp),'TN':int(tn)})
pd.DataFrame(th).to_csv(OUT/'threshold_tradeoff.csv',index=False)
summary.update({'random_split_train_test':[int(len(ytr)),int(len(yte))],'group_split_train_test':[int(len(ygtr)),int(len(ygte))],'group_split_test_class_percent':{str(k):float(v) for k,v in (ygte.value_counts(normalize=True).sort_index()*100).round(2).items()},'models_trained':list(random_df.index),'best_model_random_split':best,'best_model_metrics_random_split':{k:(int(v) if k in ['TN','FP','FN','TP'] else float(v)) for k,v in random_df.loc[best].round(4).to_dict().items()},'best_group_split_model_by_f1':group_df['F1'].idxmax(),'best_group_split_metrics':{k:(int(v) if k in ['TN','FP','FN','TP'] else float(v)) for k,v in group_df.iloc[0].round(4).to_dict().items()},'top_feature_importances':{k:float(v) for k,v in imps.head(10).round(4).items()}})
with open(OUT/'analysis_summary.json','w',encoding='utf-8') as f: json.dump(summary,f,indent=2)
print('Analysis complete')
print(random_df.round(4).to_string())
print(group_df.round(4).to_string())
