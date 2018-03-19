from SVDPP import SVDPP
import numpy as np
class TBTFD(SVDPP):
    def __init__(self,X,behavior,ad_cate,k=50):
        super().__init__(X,k)
        self.behavior=behavior#behavior[uid][cateid]=btag,    behavior.csv
        self.ad_cate=ad_cate#adid=>cateid                    ad_feature.csv
        '''self.uave={}
        self.adave={}
        self.uNum={}
        self.adNum={}
        for i in range(self.X.shape[0]):
            uid=self.X[i,0]
            adid=self.X[i,1]
            clk=self.X[i,2]
            self.uave.setdefault(uid, 0)
            self.adave.setdefault(adid, 0)
            self.uNum.setdefault(uid,0)
            self.adNum.setdefault(adid,0)
            self.uNum[uid]+=1
            self.adNum[adid]+=1
            if clk==1:
                self.uave[uid]+=1
                self.adave[adid]+=1'''
        pass

    def pred(self,uid,adid):

        #print("TBSVD pred!!!")
        size=self.X.shape[0]
        '''self.uave.setdefault(uid,0)
        self.adave.setdefault(adid,0)
        self.uNum.setdefault(uid,1)
        self.adNum.setdefault(adid,1)'''
        self.bi.setdefault(adid, 0)
        self.bu.setdefault(uid, 0)
        self.qi.setdefault(adid, np.zeros((self.k, 1)))
        self.pu.setdefault(uid, np.zeros((self.k, 1)))
        self.yi.setdefault(adid, np.zeros((self.k, 1)))
        if self.qi[adid].all() == None:
            self.qi[adid] = np.zeros((self.k, 1))
        if self.yi[adid].all() == None:
            self.yi[adid] = np.zeros((self.k, 1))
        if self.pu[uid].all() == None:
            self.pu[uid] = np.zeros((self.k, 1))
        z = np.sum(self.yi[adid]) / np.sqrt(self.ru)

        if adid not in self.ad_cate or uid not in self.behavior:
            be=0.0
        else:
            #self.behavior.setdefault(uid,0)
            cateid=self.ad_cate[adid]#异常处理
            be = self.behavior[uid][cateid] if cateid in self.behavior[uid] else 0.0
        #ano=self.uave[uid]/self.uNum[uid]
        ans =self.bi[adid] + self.bu[uid] + np.sum(self.qi[adid] * (self.pu[uid] + z))   #self.behavior[uid][cateid]
        print("ans:",ans)
        ans+=self.ave*(1+be) if be>0.1 else 0
        if ans > 1:
            return 1
        if ans < 0:
            return 0
        return ans
        pass

    def train(self,steps=20,gamma=0.04,Lambda=0.15,Lambda1=0.15):
        #加交叉验证防止过拟合
        '''cv_train=[]
        cv_test=[]
        minRmse=1'''

        for step in range(steps):
            #print("the", step,"-th step is running")
            rmse_sum=0.0
            kk=np.random.permutation(self.X.shape[0])#返回洗牌副本，shuffle打乱原本
            for j in range(1000):#self.X.shape[0]
                i=kk[j]
                uid=self.X[i][0]
                adid=self.X[i][1]
                click=self.X[i][2]
                eui=click-self.pred(uid,adid)
                rmse_sum+=eui**2
                self.bu[uid]+=gamma*(eui-Lambda*self.bu[uid])
                self.bi[adid]+=gamma*(eui-Lambda*self.bi[adid])
                tmp=self.qi[adid]
                z = np.sum(self.yi[adid]) / np.sqrt(self.ru)
                self.qi[adid]+=gamma*(eui*(self.pu[uid]+z)-Lambda1*self.qi[adid])
                self.pu[uid]+=gamma*(eui*tmp-Lambda1*self.pu[uid])
                print("pu:",self.pu[uid])
                self.yi[adid]+=gamma*(eui*self.ru*self.qi[adid]-Lambda1*self.yi[adid])
                print("yi:",self.yi[adid])
            gamma*=0.93
            #print("TBSVD module:the rmse of this step on train data is:",np.sqrt(rmse_sum/self.X.shape[0]))
            '''curRmse=self.cal_rmse(cv_test)
            if curRmse-minRmse>0.01:
                print("the best step is:",step)
                break'''
        pass



