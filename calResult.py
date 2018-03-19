import numpy as np
from sklearn.metrics import roc_auc_score

class calResult:
    def __init__(self):
        pass
    '''
    1)	Rmse：方俊根差
    2)	Auc：roc曲线下面积
    3)	Fscore：pre和recall的折中
    4)	Recall：正例查全率
    5)	Pre：正例精确度
    6)	Acc：整体精确度
    7)	Time
    '''
    def gaussMarkOpt(self,score,base=0.04):#修改！！！！
        if score<base:
            return 0
        else:
            return 1

    def cal_rmse(self,output,test_X ):
        sums=0
        '''for i in range(len(output)):
            output[i]=self.gaussMarkOpt(output[i],base)'''
        test_X=np.array(test_X)
        '''t=time.time()
        for i in range(test_X.shape[0]):
            pre=self.pred(test_X[i][0],test_X[i][1])
            sums+=(pre-test_X[i][2])**2
        rmse = np.sqrt(sums / test_X.shape[0])
        print("normal cost:",time.time()-t)
        print("the rmse on test data is:", rmse)'''

        #t=time.time()
        '''pool=ThreadPool(16)
        #print(test_X.shape[0])
        uid=test_X[:,0]
        adid=test_X[:,1]
        tasks=zip(uid,adid)
        pre=pool.starmap(self.pred,tasks)#map对接函数只接受一个参数，starmap可接受多个
        pool.close()
        pool.join()
        self.output=pre'''
        sums=np.sum((output-test_X[:,3])**2)
        rmse=np.sqrt(sums/test_X.shape[0])
        #print("thread cost time:",time.time()-t)
        #print("the svd module of rmse on test data is:",rmse)#0.236766636577
        return rmse
        pass

    def cal_acc(self,output,test_X):#精确度,正确分类个数/总个数
        test_X = np.array(test_X)
        count=0
        sz=len(output)
        if sz==0:
            print("验证集大小为0！！！")
            return 0
        for i in range(sz):
            if output[i]==test_X[i,3]:
                count+=1
        acc=count/sz
        return acc
        pass

    def cal_pre(self,output,test_X):#查准率,正例类别分类的准确度,正例正确分类个数/分类为正例的个数
        test_X = np.array(test_X)
        sz=len(output)
        posNum=0
        rightPosNum=0
        for i in range(sz):
            if output[i]==1:
                posNum+=1
                if test_X[i,3]==1:
                    rightPosNum+=1
        if posNum==0:
            print("分类为正例个数为0！！！")
            return 0
        pre=rightPosNum/posNum
        return pre
        pass

    def cal_recall(self,output,test_X):#查全率,正例正确分类个数/实际正例个数
        test_X = np.array(test_X)
        sz = len(output)
        posNum = 0
        rightPosNum = 0
        for i in range(sz):
            if test_X[i,3] == 1:
                posNum += 1
                if output[i] == 1:
                    rightPosNum += 1
        if posNum == 0:
            print("实际正例个数为0！！！")
            return 0
        recall = rightPosNum / posNum
        return recall
        pass

    def cal_Fscore(self,output,test_X):#F-score，由于查准率和查全率难一般相互抑制，fscore是查准率和查全率的调和平均值,接近查准率和查全率中较小的那个，要是fscore很高，查准和查全都必须很高
        pre=self.cal_pre(output,test_X)
        recall=self.cal_recall(output,test_X)
        if pre==0 or recall==0:
            print("查准率或者查全率为0！！！")
            return 0
        return 2*pre*recall/(pre+recall)
        pass

    def cal_auc(self,output,test_X):#auc,纵轴为真阳率，横轴为假阳率的曲线下的面积
        test_X=np.array(test_X)
        auc=roc_auc_score(test_X[:,3],output)
        return auc
        pass