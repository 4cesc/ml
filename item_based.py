import numpy as np
from calResult import calResult
class item_based(calResult):
    def __init__(self,X):
        self.X=np.array(X)
        print("the input data size is:",self.X.shape)
        self.user_ad={}
        self.ad_user={}
        self.ave=np.mean(self.X[:,2])
        print("ave:",self.ave)
        for i in range(self.X.shape[0]):
            uid=self.X[i][0]
            adid=self.X[i][1]
            click=self.X[i][2]
            self.user_ad.setdefault(uid,{})
            self.ad_user.setdefault(adid,{})
            self.user_ad[uid][adid]=click
            self.ad_user[adid][uid]=click
        self.similarity={}
        pass

    def sim_cal(self,ad1,ad2):
        self.similarity.setdefault(ad1,{})
        self.similarity.setdefault(ad2,{})
        self.ad_user.setdefault(ad1,{})
        self.ad_user.setdefault(ad2,{})
        self.similarity[ad1].setdefault(ad2,-1)
        self.similarity[ad2].setdefault(ad1,-1)
        if self.similarity[ad1][ad2]!=-1:
            return self.similarity[ad1][ad2]
        si={}
        for user in self.ad_user[ad1]:
            if user in self.ad_user[ad2]:
                si[user]=1
        n=len(si)
        if n==0:
            self.similarity[ad1][ad2]=0
            self.similarity[ad2][ad1]=0
            return 0
        s1=np.array([self.ad_user[ad1][u] for u in si])
        s2=np.array([self.ad_user[ad2][u] for u in si])
        sum1=np.sum(s1)
        sum2=np.sum(s2)
        sum1Sq=np.sum(s1**2)
        sum2Sq=np.sum(s2**2)
        pSum=np.sum(s1*s2)
        num=pSum-(sum1*sum2/n)
        den=np.sqrt((sum1Sq-sum1**2/n)*(sum2Sq-sum2**2/n))
        if den==0:
            self.similarity[ad1][ad2]=0
            self.similarity[ad2][ad1]=0
            return 0
        self.similarity[ad1][ad2]=num/den
        self.similarity[ad2][ad2]=num/den
        return num/den
        pass

    def pred(self,uid,adid):
        sim_accumulate=0.0
        click_acc=0.0
        if uid not in self.user_ad:
            return self.ave
        for item in self.user_ad[uid]:
            sim=self.sim_cal(item,adid)
            if sim<0:
                continue
            click_acc+=sim*self.user_ad[uid][item]
            sim_accumulate+=sim
        if sim_accumulate==0:
            return self.ave
        return click_acc/sim_accumulate
        pass

    def cal_output(self,test_X):
        output=[]
        test_X=np.array(test_X)
        for i in range(test_X.shape[0]):
            pre=self.pred(test_X[i][0],test_X[i][1])
            output.append(pre)
        return output



    '''def cal_rmse(self,test_X):
        test_X=np.array(test_X)
        sums=0
        print("the test data size is:",test_X.shape)
        #f=open('E:/result.txt','w')
        for i in range(test_X.shape[0]):
            pre=self.pred(test_X[i][0],test_X[i][1])
            sums+=(pre-test_X[i][2])**2
            #f.write(str(pre)+"\t"+str(test_X[i][2])+"\n")
        #f.close()
        print("result write down!!!")
        rmse=np.sqrt(sums/test_X.shape[0])
        #print("the item_based modle of rmse on test data is:",rmse)
        return rmse
        pass'''

    #0.220209123814





