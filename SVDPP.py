from SVD import SVD_C
import numpy as np
from numpy.random import random
from calResult import calResult
from matplotlib import pylab as pl
class SVDPP(calResult):
    def __init__(self,X,k=20,test_X=None):
        self.test_X=test_X
        self.output=[]
        self.X=np.array(X)
        self.k=k
        self.ave=np.mean(self.X[:,2])#统计各个user点击率，各个广告点击率
        #print("the input data size is:",self.X.shape)
        self.bi={}
        self.bu={}
        self.bp={}#新增
        self.pu={}
        self.qv={}
        self.rw={}#新增
        self.sx={}#新增brand
        '''self.ad_user={}
        self.user_price={}
        self.price_ad={}
        self.user_brand = {}#新增
        self.ad_brand = {}'''

        self.user_clk={}
        self.ad_clk={}
        self.brand_clk={}
        for i in range(self.X.shape[0]):
            uid=self.X[i][0]
            adid=self.X[i][1]
            price = self.X[i][2]
            click = self.X[i][3]
            brand=self.X[i][4]#新增
            '''self.user_price.setdefault(uid,{})
            self.ad_user.setdefault(adid,{})
            self.price_ad.setdefault(price, {})
            self.ad_brand.setdefault(adid, {})
            self.user_brand.setdefault(uid, {})'''

            self.user_clk.setdefault(uid,{})#统计用户平均点击率
            if click in self.user_clk[uid]:
                self.user_clk[uid][click]+=1
            else:
                self.user_clk[uid][click] = 1

            self.ad_clk.setdefault(adid, {})#统计商品平均点击率
            if click in self.ad_clk[adid]:
                self.ad_clk[adid][click] += 1
            else:
                self.ad_clk[adid][click] = 1

            self.brand_clk.setdefault(brand, {})  #统计品牌平均点击率
            if click in self.brand_clk[brand]:
                self.brand_clk[brand][click]+=1
            else:
                self.brand_clk[brand][click]=1

            '''self.ad_user[adid][uid]=click
            self.user_price[uid][price]=click
            self.price_ad[price][adid]=click
            self.ad_brand[adid][brand]=click#新增
            self.user_brand[uid][brand]=click'''
            self.bi.setdefault(adid,0)
            self.bu.setdefault(uid,0)
            self.bp.setdefault(price,0)
            #self.qv.setdefault(adid, random((self.k, 1))  / 10 * (np.sqrt(self.k)))
            #self.pu.setdefault(uid,  random((self.k, 1))  / 10 * (np.sqrt(self.k)))
            self.qv.setdefault(adid,(2*random((self.k,1))-1)/10*(np.sqrt(self.k)))
            self.pu.setdefault(uid,(2*random((self.k,1))-1)/10*(np.sqrt(self.k)))
            self.rw.setdefault(price, (2 * random((self.k, 1)) - 1) / 10 * (np.sqrt(self.k)))
        self.userAve={}
        for uid in self.user_clk:
            if 1 not in self.user_clk[uid]:
                self.userAve[uid]=0
            elif 0 not in self.user_clk[uid]:
                self.userAve[uid] = 1
            else:
                self.userAve[uid]=self.user_clk[uid][1]/(self.user_clk[uid][1]+self.user_clk[uid][0])
        self.adAve = {}
        for adid in self.ad_clk:
            if 1 not in self.ad_clk[adid]:
                self.adAve[adid] = 0
            elif 0 not in self.ad_clk[adid]:
                self.adAve[adid] = 1
            else:
                self.adAve[adid] = self.ad_clk[adid][1] / (self.ad_clk[adid][1] + self.ad_clk[adid][0])
        self.user_btag={}
        self.btag={}
        self.btag[2]=0
        self.btag[3]=0
        self.btag[4]=0
        self.count = {}
        self.count[2] = 0
        self.count[3] = 0
        self.count[4] = 0
        with open('filterPvBehavior.csv','r') as f:
            line=f.readline()
            while line:
                line=f.readline()
                if line=='':
                    break
                data=line.split(',')
                #print(data)
                uid,adid,tag=int(float(data[0])),int(float(data[5])),int(data[2])
                self.user_btag.setdefault(uid,{})
                self.user_btag[uid][adid]=tag
                self.count[tag]+=1
                if int(data[-1])==0:
                    continue
                self.btag[tag]+=1


    def pred(self,uid,adid,price):
        #print("SVD pred!!!")
        self.bi.setdefault(adid,0)
        self.bu.setdefault(uid,0)
        self.bp.setdefault(price, 0)
        self.qv.setdefault(adid,np.zeros((self.k,1)))
        self.pu.setdefault(uid,np.zeros((self.k,1)))
        self.rw.setdefault(price, np.zeros((self.k, 1)))
        if self.qv[adid].all()==None:
            self.qv[adid]=np.zeros((self.k,1))
        if self.pu[uid].all()==None:
            self.pu[uid]=np.zeros((self.k,1))
        if self.rw[price].all()==None:
            self.rw[price]=np.zeros((self.k,1))
        uave=self.userAve[uid] if uid  in self.userAve else self.ave
        adave = self.adAve[adid] if adid in self.adAve else 0
        tag=1
        if uid in self.user_btag and adid in self.user_btag[uid]:
            tag=self.user_btag[uid][adid]
            #self.btag[tag] / self.count[tag]
        ans=uave+self.bi[adid]+self.bu[uid]+self.bp[price]+np.sum(self.qv[adid]*self.pu[uid])+np.sum(self.qv[adid]*self.rw[price])+np.sum(self.rw[price]*self.pu[uid])
        if tag==2:
            ans*=(61356/62248)
        elif tag==3:
            ans*=(63703/62248)
        elif tag==4:
            ans*=(55785/62248)
        else:
            ans*=(62328/62248)
        if ans>1:
            return 1
        if ans<0:
            return 0
        return ans
        pass

    def train(self,steps=10 ,gamma=0.04,Lambda=0.15):
        aucList = []
        rmseList = []
        for step in range(steps):
            rmse_sum=0.0
            kk=np.random.permutation(self.X.shape[0])
            for j in range(self.X.shape[0]):#
                i=kk[j]
                uid=self.X[i][0]
                adid=self.X[i][1]
                price=self.X[i][2]
                click=self.X[i][3]
                eui=click-self.pred(uid,adid,price)
                rmse_sum+=eui**2
                self.bu[uid]+=gamma*(eui-Lambda*self.bu[uid])
                self.bi[adid]+=gamma*(eui-Lambda*self.bi[adid])
                self.bp[price] += gamma * (eui - Lambda * self.bp[price])
                tmpQv=self.qv[adid]
                tmpPu = self.pu[uid]
                tmpRw = self.rw[price]
                self.pu[uid]+=gamma*(eui*(tmpQv+tmpRw)-Lambda*self.pu[uid])
                self.qv[adid] += gamma * (eui *(tmpPu+tmpRw) - Lambda * self.qv[adid])
                self.rw[price] += gamma * (eui * ( tmpPu+tmpQv) - Lambda * self.rw[price])
            gamma*=0.93
            print("the rmse of this step on train data is:",np.sqrt(rmse_sum/self.X.shape[0]))
        print("train done!!!")

    def cal_output(self,test_X):
        output=[]
        test_X=np.array(test_X)
        for i in range(test_X.shape[0]):
            pre=self.pred(test_X[i][0],test_X[i][1],test_X[i][2])
            output.append(pre)
        return output




