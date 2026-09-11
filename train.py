"""Reproducible synthetic benchmark; early inputs predict a later simulated outcome."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss, confusion_matrix, roc_curve
from sklearn.calibration import calibration_curve

ROOT=Path(__file__).resolve().parent
FEATURES=['attendance_pct','submission_pct','feedback_score','mentor_sessions']
def run():
    rng=np.random.default_rng(42)
    n=1500
    engagement=rng.normal(0,1,n)
    x=np.column_stack([np.clip(78+14*engagement+rng.normal(0,10,n),20,100),np.clip(72+16*engagement+rng.normal(0,14,n),0,100),np.clip(3.4+.5*engagement+rng.normal(0,.7,n),1,5),rng.integers(0,9,n)])
    x=np.round(x,2)
    z=.045*(x[:,0]-75)+.04*(x[:,1]-70)+.7*(x[:,2]-3)+.13*(x[:,3]-3)+rng.normal(0,.7,n)
    y=rng.binomial(1,1/(1+np.exp(-z)))
    df=pd.DataFrame(x,columns=FEATURES)
    df.insert(0,'intern_id',[f'INT-{i+1:04}' for i in range(n)])
    df['department']=rng.choice(['Data Analytics','Web Development','Design','Marketing'],n)
    df['final_success']=y
    train,test=train_test_split(np.arange(n),test_size=.25,stratify=y,random_state=42)
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=42))
    cv=cross_val_score(model,x[train],y[train],cv=5,scoring='roc_auc')
    model.fit(x[train],y[train]); p=model.predict_proba(x[test])[:,1]
    dummy=DummyClassifier(strategy='prior').fit(x[train],y[train]); base=dummy.predict_proba(x[test])[:,1]
    pred=p>=.5
    metrics={'accuracy':accuracy_score(y[test],pred),'precision':precision_score(y[test],pred),'recall':recall_score(y[test],pred),'f1':f1_score(y[test],pred),'roc_auc':roc_auc_score(y[test],p),'brier':brier_score_loss(y[test],p),'baseline_brier':brier_score_loss(y[test],base),'baseline_auc':roc_auc_score(y[test],base),'cv_auc_mean':cv.mean(),'cv_auc_std':cv.std(),'train_count':len(train),'test_count':len(test)}
    scale,lr=model.steps[0][1],model.steps[1][1]
    export={'features':FEATURES,'mean':scale.mean_.tolist(),'scale':scale.scale_.tolist(),'coef':lr.coef_[0].tolist(),'intercept':float(lr.intercept_[0])}
    portable=1/(1+np.exp(-(((x-scale.mean_)/scale.scale_)@lr.coef_[0]+lr.intercept_[0])))
    assert np.max(np.abs(portable-model.predict_proba(x)[:,1]))<1e-12
    df['split']='train';df.loc[test,'split']='test';df['success_probability']=portable
    assert df.intern_id.is_unique and not df.isna().any().any()
    for d in ['data','dist','reports']: (ROOT/d).mkdir(exist_ok=True)
    df.to_csv(ROOT/'data/interns.csv',index=False)
    df.iloc[test].to_csv(ROOT/'reports/test_predictions.csv',index=False)
    fpr,tpr,_=roc_curve(y[test],p); actual,estimated=calibration_curve(y[test],p,n_bins=8)
    payload={'model':export,'metrics':metrics,'records':df.to_dict('records'),'confusion':confusion_matrix(y[test],pred).tolist(),'roc':list(zip(fpr,tpr)),'calibration':list(zip(estimated,actual))}
    (ROOT/'dist/bundle.js').write_text('const BUNDLE = '+json.dumps(payload)+';')
    (ROOT/'reports/model.json').write_text(json.dumps(export,indent=2))
    (ROOT/'reports/metrics.json').write_text(json.dumps(metrics,indent=2))
    plt.style.use('default');plt.rcParams.update({'figure.facecolor':'#f1f6fb','axes.facecolor':'white','text.color':'#18364d','axes.labelcolor':'#18364d','xtick.color':'#566f84','ytick.color':'#566f84','axes.spines.top':False,'axes.spines.right':False});fig,ax=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    fig.suptitle('Intern Performance Prediction | Synthetic held-out evaluation',fontsize=17)
    ax[0,0].plot(fpr,tpr,color='#0872bc',label=f'AUC {metrics["roc_auc"]:.3f}');ax[0,0].plot([0,1],[0,1],'--',color='#ed8a36');ax[0,0].set(title='ROC curve',xlabel='False positive rate',ylabel='True positive rate');ax[0,0].legend()
    ax[0,1].plot(estimated,actual,'o-',color='#0872bc');ax[0,1].plot([0,1],[0,1],'--',color='#ed8a36');ax[0,1].set(title='Calibration',xlabel='Mean predicted probability',ylabel='Observed success fraction')
    cm=payload['confusion'];ax[1,0].imshow(cm,cmap='Blues');ax[1,0].set(title='Confusion matrix at 0.50',xlabel='Predicted class',ylabel='Actual class',xticks=[0,1],yticks=[0,1])
    for i in range(2):
        for j in range(2):ax[1,0].text(j,i,str(cm[i][j]),ha='center',color='orange',fontsize=22)
    ax[1,1].barh(FEATURES,lr.coef_[0],color='#0872bc');ax[1,1].set(title='Standardized coefficients (association)',xlabel='Log-odds coefficient')
    fig.savefig(ROOT/'dist/model_evaluation.png',dpi=160);plt.close(fig)
    print(json.dumps(metrics,indent=2)); print('PASS: data integrity and exported-model numerical parity')
if __name__=='__main__':run()
