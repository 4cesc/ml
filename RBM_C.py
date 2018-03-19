import numpy as np
from numpy.random import normal,random,uniform
class TEMP:
    def __init__(self):
        self.AccVH=None
        self.CountVH=None
        self.AccV=None
        self.temp.CountV=None
        self.temp.AccH=None

class CF_RBM:
    def __init__(self,X,UserNum=384559*2,HiddenNum=30,ItemNum=100,click=1):
        self.X=np.array(X)
        self.HiddenNum=HiddenNum
        self.ItemNum=ItemNum
        self.UserNum=UserNum
        self.click=click
        self.ad_user={}
        self.user_ad={}
        self.bik=np.zeros((self.ItemNum,self.click))
        self.Momentum={}#动量
        self.Momentum['bik']=np.zeros((self.ItemNum,self.click))
        self.UMatrix=np.zeros((self.UserNum,self.ItemNum))
        self.V=np.zeros((self.ItemNum,self.click))
        for i in range(self.X.shape[0]):
            uid=self.X[i][0]-1
            adid=self.X[i][1]-1
            click=self.X[i][2]-1
            self.UMatrix[uid][adid]=1
            self.bik[adid][click]+=1
            self.ad_user.setdefault(adid,{})
            self.user_ad.setdefault(uid,{})
            self.ad_user[adid][uid]=click
            self.user_ad[uid][adid]=click
        self.W=normal(0,0.01,(self.ItemNum,self.click,HiddenNum))
        self.Momentum['W']=np.zeros(self.W.shape)
        self.initialize_bik()
        self.bj=np.zeros((HiddenNum,1)).flatten(1)
        self.Momentum['bj']=np.zeros(self.bj.shape)
        self.Dij=np.zeros((self.ItemNum,self.HiddenNum))
        self.Momentum['Dij']=np.zeros((self.ItemNum,self.HiddenNum))

    def initialize_bik(self):
        for i in range(self.ItemNum):
            total=np.sum(self.bik[i])
            if total>0:
                for k in range(self.click):
                    if self.bik[i][k]==0:
                        self.bik[i][k]=-10
                    else:
                        self.bik[i][k]=np.log(self.bik[i][k]/total)

    def test(self,test_X):
        output=[]
        sums=0
        test_X=np.array(test_X)
        for i in range(test_X.shape[0]):
            pre=self.pred(test_X[i][0]-1,test_X[i][1]-1)
            output.append(pre)
            sum+=(pre-test_X[i][2])**2
        rmse=np.sqrt(sums/test_X.shape[0])
        print("the rmse on test data is:",rmse)
        return output

    def pred(self,uid,adid):
        V=self.clamp_user(uid)
        pj=self.update_hidden(V,uid)
        vp=self.update_visible(pj,uid,adid)
        ans=0
        for i in range(self.click):
            ans+=vp[i]*(i+1)
        return ans

    def clamp_user(self,uid):
        V=np.zeros(self.V.shape)
        for i in self.user_ad[uid]:
            V[i][self.user_ad[uid][i]]=1
        return V

    def train(self,para,test_X,cd_steps=3,batch_size=30,numEpoch=100,Err=0.00001):
        for epo in range(numEpoch):
            print("the ",epo,"-th epoch is running")
            kk=np.random.permutation(range(self.UserNum))
            bt_count=0
            while bt_count<=self.UserNum:
                btend=min(self.UserNum,bt_count+batch_size)
                users=kk[bt_count:btend]
                temp=TEMP
                temp.AccVH=np.zeros(self.W.shape)
                temp.CountVH=np.zeros(self.W.shape)
                temp.AccV=np.zeros(self.V.shape)
                temp.CountV=np.zeros(self.V.shape)
                temp.AccH=np.zeros(self.bj.shape)
                watched=np.zeros(self.UMatrix[0].shape)
                for user in users:
                    watched[self.UMatrix[user]==1]=1
                    sv=self.clamp_user(user)
                    pj=self.update_hidden(sv,user)
                    temp=self.accum_temp(sv,pj,temp,user)
                    for step in range(cd_steps):
                        sh=self.sample_hidden(pj)
                        vp=self.update_visible(sh,user)
                        sv=self.sample_visible(vp,user)
                    deaccum_temp=self.deaccum_temp(sv,pj,temp,user)
                self.updateall(temp,batch_size,para,watched)
                bt_count+=batch_size
            self.test(test_X)

    def accum_temp(self,V,pj,temp,uid):
        for i in self.user_ad[uid]:
            temp.AccVH[i]+=np.dot(V[i].reshape(-1,1),pj.reshape(1,-1))
            temp.CountVH[i]+=1
            temp.AccV[i]+=V[i]
            temp.CountVH[i]+=1
        temp.AccH+=pj
        return temp

    def deaccum_temp(self,V,pj,temp,uid):
        for i in self.user_ad[uid]:
            temp.AccVH[i]-=np.dot(V[i].reshape(-1,1),pj.reshape(1,-1))
            temp.AccV[i]-=V[i]
        temp.AccH-=pj
        return temp

    def updateall(self,temp,batch_size,para,watched):
        delatW=np.zeros(temp.CountVH.shape)
        delatBik=np.zeros(temp.CountV.shape)
        delatW[temp.CountVH!=0]=temp.AccVH[temp.CountVH!=0]/temp.CountVH[temp.CountVH!=0]
        delatBik[temp.CountV!=0]=temp.AccV[temp.CountV!=0]/temp.CountV[temp.CountV!=0]
        delataBj=temp.AccH/batch_size

        self.Momentum['W'][temp.CountVH!=0]=self.Momentum['W'][temp.CountVH!=0]*para['Momentum']
        self.Momentum['W'][temp.CountVH!=0]+=para['W']*(delatW[temp.CountVH!=0]-para['weight_cost']*self.W[temp.CountVH!=0])
        self.W[temp.CountVH!=0]+=self.Momentum['W'][temp.CountVH!=0]

        self.Momentum['bik'][temp.CountV!=0]=self.Momentum['bik'][temp.CountV!=0]*para['Momentum']
        self.Momentum['bik'][temp.CountV!=0]+=para['bik']*delatBik[temp.CountV!=0]
        self.bik[temp.CountV!=0]+=self.Momentum['bik'][temp.CountV!=0]

        self.Momentum['bj']=self.Momentum['bj']*para['Momentum']
        self.Momentum['bj']+=para['bj']*delataBj
        self.bj+=self.Momentum['bj']

        for i in range(self.ItemNum):
            if watched[i]==1:
                self.Momentum['Dij'][i]=self.Momentum['Dij'][i]*para['Momentum']
                self.Momentum['Dij'][i]+=para['D']*temp.AccH/batch_size
                self.Dij[i]+=self.Momentum['Dij'][i]

    np.seterr(all='raise')

    def update_hidden(self,V,uid):
        r=self.UMatrix[uid]
        hp=None
        count=1
        for i in self.user_ad[uid]:
            if count==1:
                hp=np.dot(V[i],self.W[i]).flatten(1)
                count+=1
            else:
                hp+=np.dot(V[i],self.W[i]).flatten(1)
        pj=1/(1+np.exp(-self.bj-hp+np.dot(r,self.Dij).flatten(1)))
        return pj

    def sample_hidden(self,pj):
        sh=uniform(size=pj.shape)
        for i in range(sh.shape[0]):
            if sh[i]<pj[i]:
                sh[i]=1.0
            else:
                sh[i]=0.0
        return sh

    def update_visible(self,sh,uid,adid=None):
        if adid==None:
            vp=np.zeros(self.V.shape)
            for i in self.user_ad[uid]:
                vp[i]=np.exp(self.bik[i]+np.dot(self.W[i],sh))
                vp[i]=vp[i]/np.sum(vp[i])
            return vp
        vp=np.exp(self.bik[adid]+np.dot(self.W[adid],sh))
        vp=vp/np.sum(vp)
        return vp

    def sample_visible(self,vp,uid):
        sv=np.zeros(self.V.shape)
        for i in self.user_ad[uid]:
            r=uniform()
            k=0
            for k in range(self.click):
                r-=vp[i][k]
                if r<0:
                    break
                sv[i][k]=1
        return sv