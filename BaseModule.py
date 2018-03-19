import numpy as np
import simplejson as json
from numpy.random import random
import time
from multiprocessing.dummy import Pool as ThreadPool
from calResult import calResult
class BaseModule(calResult):
    def __init__(self,X,k=20):
        self.output=[]
        self.X=np.array(X)
        self.k=k
        self.ave=np.mean(self.X[:,2])
        #print("the input data size is:",self.X.shape)
        self.bi={}
        self.bu={}
        self.ad_user={}
        self.user_ad={}
        for i in range(self.X.shape[0]):
            uid=self.X[i][0]
            adid=self.X[i][1]
            click=self.X[i][2]
            self.user_ad.setdefault(uid,{})
            self.ad_user.setdefault(adid,{})
            self.ad_user[adid][uid]=click
            self.user_ad[uid][adid]=click
            self.bi.setdefault(adid,0)
            self.bu.setdefault(uid,0)
            #self.qi.setdefault(adid, random((self.k, 1))  / 10 * (np.sqrt(self.k)))
            #self.pu.setdefault(uid,  random((self.k, 1))  / 10 * (np.sqrt(self.k)))
            #self.qi.setdefault(adid,(2*random((self.k,1))-1)/10*(np.sqrt(self.k)))
            #self.pu.setdefault(uid,(2*random((self.k,1))-1)/10*(np.sqrt(self.k)))
            #print(self.qi)
            #print(self.pu)

    def pred(self,uid,adid):
        #print("SVD pred!!!")
        self.bi.setdefault(adid,0)
        self.bu.setdefault(uid,0)
        #self.qi.setdefault(adid,np.zeros((self.k,1)))
        #self.pu.setdefault(uid,np.zeros((self.k,1)))
        '''if self.qi[adid].all()==None:
            self.qi[adid]=np.zeros((self.k,1))
        if self.pu[uid].all()==None:
            self.pu[uid]=np.zeros((self.k,1))'''
        ans=self.ave+self.bi[adid]+self.bu[uid]
        if ans>1:
            return 1
        if ans<0:
            return 0
        return ans
        pass

    def train(self,steps=500 ,gamma=0.04,Lambda=0.15):
        for step in range(steps):
            print("the", step,"-th step is running")
            rmse_sum=0.0
            kk=np.random.permutation(self.X.shape[0])
            for j in range(10000):#self.X.shape[0]
                i=kk[j]
                uid=self.X[i][0]
                adid=self.X[i][1]
                click=self.X[i][2]
                eui=click-self.pred(uid,adid)
                rmse_sum+=eui**2
                self.bu[uid]+=gamma*(eui-Lambda*self.bu[uid])
                self.bi[adid]+=gamma*(eui-Lambda*self.bi[adid])
            gamma*=0.93
            print("the rmse of this step on train data is:",np.sqrt(rmse_sum/self.X.shape[0]))
        pass

    def cal_output(self,test_X):
        output=[]
        test_X=np.array(test_X)
        f=open('BaseModuleData.txt','w+')
        for i in range(test_X.shape[0]):
            pre=self.pred(test_X[i][0],test_X[i][1])
            output.append(pre)
            f.write(str(pre)+','+str(test_X[i][2])+'\n')
        f.close()
        return output

    '''def cal_rmse(self,test_X):
        sums=0
        test_X=np.array(test_X)

        #t=time.time()
        pool=ThreadPool(16)
        #print(test_X.shape[0])
        uid=test_X[:,0]
        adid=test_X[:,1]
        tasks=zip(uid,adid)
        pre=pool.starmap(self.pred,tasks)#map对接函数只接受一个参数，starmap可接受多个
        pool.close()
        pool.join()
        self.output=pre
        sums=np.sum((pre-test_X[:,2])**2)
        rmse=np.sqrt(sums/test_X.shape[0])
        #print("thread cost time:",time.time()-t)
        #print("the svd module of rmse on test data is:",rmse)#0.236766636577
        return rmse
        pass'''





