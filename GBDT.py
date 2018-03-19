import pandas as pd
import numpy as np
from sklearn.ensemble import  GradientBoostingClassifier as gbc
from sklearn import cross_validation,metrics
from sklearn.grid_search import GridSearchCV
from matplotlib import pylab as plt
from matplotlib.pylab import rcParams
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction import  *
#rcParams['figure.figsize'] = 12, 4
class GBDT:
    def __init__(self):
        pass

    def modelfit(self,alg,dtrain,predictors,performCV=True,printFeatureImportance=True,cv_folds=5):
        alg.fit(dtrain[predictors],dtrain['clk'])

        dtrain_predictions=alg.predict(dtrain[predictors])
        dtrain_predprob=alg.predict_proba(dtrain[predictors])[:,1]

        if performCV:
            cv_score=cross_validation.cross_val_score(alg,dtrain[predictors],dtrain['clk'],cv=cv_folds,scoring='roc_auc')

        print("Accuracy:",metrics.accuracy_score(dtrain['clk'].values, dtrain_predictions))
        print("AUC score (train):",metrics.roc_auc_score(dtrain['clk'], dtrain_predprob))

        if printFeatureImportance:
            print("printFeature in!!!")
            feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
            feat_imp.plot(kind='bar', title='Feature Importances')
            plt.ylabel('Feature Importance Score')
            plt.show(feat_imp.all())
            a=input()
            plt.savefig('feature_importance.png')




if __name__=='__main__':
    train = pd.read_csv('data.csv')
    dff=pd.DataFrame(train)
    dff.fillna(dff.mean(),inplace=True)
    #pd.Series(dff).to_csv("processData.csv")
    #print(dff)
    dff.to_csv('fillData.csv')
    target = 'clk'
    predictors = [x for x in dff.columns if x not in [target]]
    gbm0 = gbc(random_state=10)
    GBDT().modelfit(gbm0, train, predictors)
    #SelectFromModel(gbc()).fit_transform(train[predictors],train[target])
    '''from sklearn.decomposition import PCA 
    
    from sklearn.decomposition import LatentDirichletAllocation as LDA
    print(PCA(n_components=2).fit_transform(train[predictors],train[target]))
    print(LDA(n_components=2).fit_transform(train[predictors],train[target]))'''
    #gsearch1.fit(train[predictors], train[target])
    #print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
#gbm0=gbc(random_state=10)
#GBDT().modelfit(gbm0,train,predictors)

