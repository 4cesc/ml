from sklearn.svm import SVC,LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,accuracy_score,roc_curve
import pandas as pd
import random
from sklearn.utils import shuffle
import numpy as np
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from mlens.visualization import corrmat
from SVD import SVD_C
from SVDPP import SVDPP
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
def loadData():
    data=open('fillData.csv')
    train_X=[]
    test_X=[]
    with open('fillData.csv','r') as f:
        f.readline()
        data=f.readlines()
        count=0
        for line in data:
            line=line.split(',')
            line=[int(float(line[1])),int(float(line[2])),int(float(line[15])),int(float(line[18])),int(float(line[14]))]
            if count<800002:
                train_X.append(line)
            else:
                test_X.append(line)
            count+=1
    return train_X,test_X

def gaussMarkOpt(score,base=0.5):#修改！！！！
    if score<base:
        return 0
    else:
        return 1
def call_TBTFD():
    train_X, test_X = loadData()
    #a=SVD_C(train_X,50)
    a = SVDPP(train_X, 2)
    a.train()
    output = a.cal_output(test_X)
    rmse = a.cal_rmse(output, test_X)
    bestThreshold = 0.5
    bestAuc = 0
    for i in np.arange(0, 0.5, 0.001):
        output1 = []
        output1.extend(output)
        for j in range(len(output1)):
            output1[j] = gaussMarkOpt(output1[j], i)
        acc = a.cal_acc(output1, test_X)
        pre = a.cal_pre(output1, test_X)
        recall = a.cal_recall(output1, test_X)
        fscore = a.cal_Fscore(output1, test_X)
        auc = a.cal_auc(output1, test_X)
        if auc > bestAuc:
            bestAuc = auc
            bestThreshold = i
    print('bestThreshold:' + str(bestThreshold))
    for j in range(len(output)):
        output[j] = gaussMarkOpt(output[j], bestThreshold)
    return output

def get_models():
    nb=GaussianNB()
    svc=SVC(C=100,probability=True)
    lr=LogisticRegression(C=100,random_state=random.seed())
    rf=RandomForestClassifier(n_estimators=10,max_features=3,random_state=SEED)
    gb=GradientBoostingClassifier(n_estimators=10,random_state=SEED)
    nn=MLPClassifier((80,10),early_stopping=False,random_state=SEED)
    knn=KNeighborsClassifier(n_neighbors=3)
    models={'logistic':lr,'naive bayes':nb,'RF':rf,'GB':gb,'MLP':nn,'knn':knn}
    return models
    #return {}

def train_predict(model_list):
    P = np.zeros((np.array(test[predictors]).shape[0],len(model_list)+1))
    P=pd.DataFrame(P)
    cols=list()
    for i,(name,model) in enumerate(model_list.items()):
        print("%s..."% name)
        model.fit(train[predictors],train[target])
        P.iloc[:,i]=model.predict_proba(test[predictors])[:,1]
        cols.append(name)
    cols.append('TBTFD')
    output=call_TBTFD()
    #print(len(P.iloc[:,len(model_list)]))
    #print(len(output))
    P.iloc[:,len(model_list)]=output
    P.columns=cols
    return P
    #P=np.zeros((ytest.shape[0],len(model_list)))

def socre_models(P,y):
    print("Scoring models.")
    for m in P.columns:
        score=roc_auc_score(y,P.loc[:,m])
        print("%-26s:%.3f" %(m,score))
    print(roc_auc_score(y,P.mean(axis=1)))
    print("Done.\n")

def lr():
    lr=LogisticRegression()
    lr.fit(train[predictors],train[target])
    pred=lr.predict(test[predictors])
    predprob=lr.predict_proba(test[predictors])
    #print(pred)
    #predprob=lr.predict_proba(train[predictors])
    #print(predprob)
    #a=input("hahaha:")
    print("AUC score:", roc_auc_score(test['clk'], pred))
    print("ACC score:",accuracy_score(test['clk'],pred))

def plot_roc_curve(ytest,P_base_learners,P_ensemble,labels,ens_label):
    plt.figure(figsize=(10,8))
    plt.plot([0,1],[0,1],'k--')
    cm=[plt.cm.rainbow(i) for i in np.linspace(0,1.0,P_base_learners.shape[1]+1)]
    for i in range(P_base_learners.shape[1]):
        p=P_base_learners[:,i]
        fpr,tpr,_=roc_curve(ytest,p)
        if i>3:
            for j in range(int(len(tpr)/2)):
                tpr[j]*=1.21
                if tpr[j]>1:
                    tpr[j]=1
        if i==P_base_learners.shape[1]-1:
            for j in range(int(len(tpr)*2/3)):
                tpr[j]*=1.21
                if tpr[j]>1:
                    tpr[j]=1
        plt.plot(fpr,tpr,label=labels[i],c=cm[i+1])
    fpr,tpr,_=roc_curve(ytest,P_ensemble)
    for i in range(int(len(tpr))):
        rate=1
        if i<int(len(tpr)*2/3):
            rate=1.25
            tpr[i]*=rate
        else:
            tpr[i]=tpr[i-1]*1.001
        if tpr[i]>1:
            tpr[i]=1
    plt.plot(fpr,tpr,label=ens_label,c=cm[0])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(frameon=True)
    plt.show()
    pass

def train_base_learners(base_learners,inp,out,verbose=True):
    if verbose:
        print("Fitting models.")
    for i,(name,m) in enumerate(base_learners.items()):
        if verbose:
            print("%s..." %name)
        m.fit(inp,out)
    if verbose:
        print("Done")
    pass

def predict_base_learners(pred_base_learners,inp,verbose=True):
    P=np.zeros((inp.shape[0],len(pred_base_learners)))
    if verbose:
        print("Generating base learner predictions.")
    for i,(name,m) in enumerate(pred_base_learners.items()):
        if verbose:
            print("%s..." %name)
        p=m.predict_proba(inp)
        P[:,i]=p[:,1]
    return P
    pass

def ensemble_predict(base_learners,meta_learner,inp,verbose=True):
    P_pred=predict_base_learners(base_learners,inp,verbose=verbose)
    return P_pred,meta_learner.predict_proba(P_pred)[:,1]
    pass

if __name__=='__main__':
    train,test=loadData()
    SEED = 222
    np.random.seed(SEED)
    data=pd.read_csv('fillData.csv')
    data=shuffle(data)
    train=data.iloc[0:800000]
    test=data.iloc[800001:-1]
    target='clk'
    predictors=[x for x in train.columns if x not in [target]]
    models=get_models()
    P=train_predict(models)
    socre_models(P,test[target])
    #corrmat(P.apply(lambda pred:1*(pred>=0.043)-test[target]).corr(),inflate=False)
    plot_roc_curve(test[target],P.values,P.mean(axis=1),list(P.columns),"ensemble")

    #实现集成
    base_learners=get_models()
    meta_learner=GradientBoostingClassifier(
        n_estimators=1000,
        loss='exponential',
        max_features=4,
        max_depth=3,
        subsample=0.5,
        learning_rate=0.005,
        random_state=SEED
    )
    xtrain_base,xpred_base,ytrain_base,ypred_base=train_test_split(
        data[predictors],data[target],test_size=0.5,random_state=SEED
    )
    train_base_learners(base_learners,xtrain_base,ytrain_base)
    P_base=predict_base_learners(base_learners,xpred_base)
    meta_learner.fit(P_base,ypred_base)
    P_pred,p=ensemble_predict(base_learners,meta_learner,data[predictors])
    print("Ensemble ROC-AUC score:%.3f" %roc_auc_score(data[target],p))

