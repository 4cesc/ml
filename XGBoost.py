import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation,metrics
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
def modelfit(alg,dtrain,predictors,useTrainCV=True,cv_folds=5,early_stopping_rounds=50):
    if useTrainCV:
        xgb_param=alg.get_xgb_params()
        xgtrain=xgb.DMatrix(dtrain[predictors].values,label=dtrain[target].values)
        cvresult=xgb.cv(xgb_param,xgtrain,num_boost_round=alg.get_xgb_params()['n_estimators'],nfold=cv_folds,
                        metrics='auc',early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
        alg.fit(dtrain[predictors],dtrain['clk'],eval_metric='auc')

        dtrain_predictions=alg.predict(dtrain[predictors])
        dtrain_predprob=alg.predict_proba(dtrain[predictors])[:,1]

        print("Accuracy : %.4g" , metrics.accuracy_score(dtrain['clk'].values, dtrain_predictions))
        print("AUC Score (Train): %f" , metrics.roc_auc_score(dtrain['clk'], dtrain_predprob))

if __name__=='__main__':
    rcParams['figure.figsize'] = 12, 4
    train = pd.read_csv('raw_sample1.csv')
    target = 'clk'
    noclk = 'noclk'
    pid = 'pid'
    predictors=[x for x in train.columns if x not in [target,noclk,pid]]
    xgb1 = XGBClassifier(
        learning_rate=0.2,
        n_estimators=20,#0.673369775119
        max_depth=30,#0.581368008271
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
    modelfit(xgb1, train, predictors)

    '''param_test1={
        'max_depth':list(range(3,10,2)),#9
        'min_child_weight':list(range(1,6,2))#1
    }
    gsearch1=GridSearchCV(estimator=XGBClassifier(
        learning_rate=0.1,
        n_estimators=140,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27),
        param_grid=param_test1,
        scoring='roc_auc',
        n_jobs=4,
        iid=False,
        cv=5
    )
    gsearch1.fit(train[predictors],train[target])
    print(gsearch1.grid_scores_,gsearch1.best_params_,gsearch1.best_score_)'''