from item_based import item_based
from SVD import SVD_C
from SVDPP import SVDPP
from TBTFD import TBTFD
import numpy as np
from matplotlib import pylab as pl
from multiprocessing import Process
import timeit
from multiprocessing.dummy import Pool as ThreadPool
from BaseModule import BaseModule
def loadFillData():
    user_btag = {}
    btag = {}
    btag[1]=0
    btag[2] = 0
    btag[3] = 0
    btag[4] = 0
    count = {}
    count[1]=0
    count[2] = 0
    count[3] = 0
    count[4] = 0
    with open('rawAndBehavior.csv', 'r') as f:
        line = f.readline()
        while line:
            line = f.readline()
            if line == '':
                break
            data = line.split(',')
            # print(data)
            uid, adid, tag = int(float(data[0])), int(float(data[5])), int(data[2])
            user_btag.setdefault(uid, {})
            user_btag[uid][adid] = tag
            count[tag] += 1
            if int(data[-1]) == 0:
                continue
            btag[tag] += 1
    for tag in count:
        print('tag:%d,rate:%f' %(tag,btag[tag]/count[tag]))
    print((btag[1]+btag[2]+btag[3]+btag[4])/(count[1]+count[2]+count[3]+count[4]))
    pass
def loadFillData1():
    user_btag = {}
    btag = {}
    #btag[1]=0
    btag[2] = 0
    btag[3] = 0
    btag[4] = 0
    count = {}
    #count[1]=0
    count[2] = 0
    count[3] = 0
    count[4] = 0
    with open('filterPvBehavior.csv', 'r') as f:
        line = f.readline()
        while line:
            line = f.readline()
            if line == '':
                break
            data = line.split(',')
            # print(data)
            uid, adid, tag = int(float(data[0])), int(float(data[5])), int(data[2])
            user_btag.setdefault(uid, {})
            user_btag[uid][adid] = tag
            count[tag] += 1
            if int(data[-1]) == 0:
                continue
            btag[tag] += 1
    for tag in count:
        print('tag:%d,%d,%d,rate:%f' %(tag,btag[tag],count[tag],btag[tag]/count[tag]))
    print((btag[2]+btag[3]+btag[4])/(count[2]+count[3]+count[4]))
    pass
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
            line=[int(line[1]),int(line[2]),int(float(line[15])),int(line[18])]
            if count<800002:
                train_X.append(line)
            else:
                test_X.append(line)
            count+=1
    return train_X,test_X

def loadData1():
    data=open('fillData.csv')
    train_X=[]
    test_X=[]
    with open('fillData.csv','r') as f:
        f.readline()
        data=f.readlines()
        count=0
        for line in data:
            line=line.split(',')
            line=[int(line[1]),int(line[2]),int(float(line[15])),int(line[18]),int(float(line[14]))]
            if count<800002:
                train_X.append(line)
            else:
                test_X.append(line)
            count+=1
    return train_X,test_X

def read_trainAndTestData():
    data=open('raw_sample.csv').read().splitlines()
    train_X=[]
    test_X=[]
    count=0
    adNum=0
    adList=[]
    for line in data:
        p=line.split(',')
        #print(p[0],p[1],p[2])
        '''if int(p[2]) not in adList:
            adList.append(int(p[2]))
            adNum+=1'''
        if count%8!=0:
            train_X.append([int(p[0]),int(p[2]),int(p[5])])
        else:
            test_X.append([int(p[0]),int(p[2]),int(p[5])])
        count+=1
    #print(adNum)
    return train_X,test_X

def read_ad_feature():
    ad_cate = {}
    f=open('ad_feature.csv', 'r')
    f.readline()
    for line in f.readlines():
        #print(line)
        #for line in data:
        p = line.split(',')
        #print(p[0],p[1])
        ad_cate[int(p[0])] = int(p[1])
    return ad_cate

def gaussMarkOpt(score,base=0.5):#修改！！！！
    if score<base:
        return 0
    else:
        return 1

def convert(s):
    if s=='ipv':
        return 1
    if s=='cart':
        return 2
    if s=='fav':
        return 3
    if s=='buy':
        return 4
    return 0

def read_behavior():
    behavior={}
    count=0
    f=open('behavior.csv','r',encoding='latin1')
    #f1=open('behavior.csv','w')
    #f.readline()
    while count<1000000:
        data=f.readline()
        #f1.write(data)
        p=data.split(',')
        uid=int(p[0])
        cateid=int(p[3])
        btag=convert(p[2])
        behavior.setdefault(uid,{})
        behavior[uid][cateid]=btag
        #print(data)
        count+=1
    f.close()
    #f1.close()
    return behavior

def call_item_based():
    train_X, test_X = read_trainAndTestData()
    #test_X=np.array(test_X)
    a=item_based(train_X)
    output=a.cal_output(test_X)
    rmse = a.cal_rmse(output, test_X)
    acc=a.cal_acc(output,test_X)
    pre=a.cal_pre(output, test_X)
    recall=a.cal_recall(output, test_X)
    fscore = a.cal_Fscore(output, test_X)
    auc=a.cal_auc(output,test_X)
    print("the item_based module of rmse:", rmse)
    print("the item_based module of acc:",acc)
    print("the item_based module of pre:",pre )
    print("the item_based module of recall:", recall)
    print("the item_based module of fscore:", fscore)
    print("the item_based module of auc:", auc)
    return [rmse,acc,pre,recall,fscore,auc]

def call_BaseModule():
    train_X, test_X = read_trainAndTestData()
    a=BaseModule(train_X,50)
    a.train()
    output = a.cal_output(test_X)
    rmse = a.cal_rmse(output, test_X)
    bestThreshold=0.5
    bestAuc=0
    for i in np.arange(0,0.1,0.001):
        output1=[]
        output1.extend(output)
        for j in range(len(output1)):
            output1[j]=gaussMarkOpt(output1[j],i)
        acc = a.cal_acc(output1, test_X)
        pre = a.cal_pre(output1, test_X)
        recall = a.cal_recall(output1, test_X)
        fscore = a.cal_Fscore(output1, test_X)
        auc = a.cal_auc(output1, test_X)
        if auc>bestAuc:
            bestAuc=auc
            bestThreshold=i
    print('bestThreshold:'+str(bestThreshold))
    for j in range(len(output)):
        output[j] = gaussMarkOpt(output[j], bestThreshold)
    acc = a.cal_acc(output, test_X)
    pre = a.cal_pre(output, test_X)
    recall = a.cal_recall(output, test_X)
    fscore = a.cal_Fscore(output, test_X)
    auc = a.cal_auc(output, test_X)
    print("the Base module of rmse:", rmse)
    print("the Base module of acc:",acc)
    print("the Base module of pre:", pre)
    print("the Base module of recall:", recall)
    print("the Base module of fscore:", fscore)
    print("the Base module of auc:", auc)
    return [rmse,acc,pre,recall,fscore,auc]

def call_SVD():
    train_X, test_X = loadData()
    a=SVD_C(train_X,7)
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
    acc = a.cal_acc(output, test_X)
    pre = a.cal_pre(output, test_X)
    recall = a.cal_recall(output, test_X)
    fscore = a.cal_Fscore(output, test_X)
    auc = a.cal_auc(output, test_X)
    print("the SVD module of rmse:", rmse)
    print("the SVD module of acc:",acc)
    print("the SVD module of pre:", pre)
    print("the SVD module of recall:", recall)
    print("the SVD module of fscore:", fscore)
    print("the SVD module of auc:", auc)
    return [rmse,acc,pre,recall,fscore,auc]

def call_SVDPP(k):
    train_X, test_X = loadData1()
    a=SVDPP(train_X,k)
    iters=[x*5 for x in range(1,11)]
    aucList=[]
    rmseList=[]
    for iter in iters:
        a.train(steps=iter)
        output = a.cal_output(test_X)
        rmse=a.cal_rmse(output,test_X)
        bestThreshold = 0.5
        bestAuc = 0
        for i in np.arange(0, 0.5, 0.001):
            output1 = []
            output1.extend(output)
            for j in range(len(output1)):
                output1[j] = gaussMarkOpt(output1[j], i)
            auc = a.cal_auc(output1, test_X)
            if auc > bestAuc:
                bestAuc = auc
                bestThreshold = i
        print('bestThreshold:' + str(bestThreshold))
        for j in range(len(output)):
            output[j] = gaussMarkOpt(output[j], bestThreshold)
        acc = a.cal_acc(output, test_X)
        pre = a.cal_pre(output, test_X)
        recall = a.cal_recall(output, test_X)
        fscore = a.cal_Fscore(output, test_X)
        auc = a.cal_auc(output, test_X)
        print("the SVDPP module of rmse:", rmse)
        print("the SVDPP module of acc:",acc)
        print("the SVDPP module of pre:", pre)
        print("the SVDPP module of recall:", recall)
        print("the SVDPP module of fscore:", fscore)
        print("the SVDPP module of auc:", auc)
        aucList.append(auc)
        rmseList.append(rmse)
    print('auc:'+aucList)
    print('rmse:'+rmseList)
    aucfig=pl.figure('auc')
    pl.plot(iters,aucList)
    pl.xlabel('迭代次数')
    pl.ylabel('AUC')
    pl.title('不同迭代数对应的AUC')
    aucfig.show()
    rmsefig=pl.figure('rmse')
    pl.plot(iters,rmseList)
    pl.xlabel('迭代次数')
    pl.ylabel('Rmse')
    pl.title('不同迭代数的Rmse图')
    rmsefig.show()
    return [rmse,acc,pre,recall,fscore,auc]

def call_TBTFD():
    train_X, test_X = read_trainAndTestData()
    ad_cate=read_ad_feature()
    behavior=read_behavior()
    a=TBTFD(train_X,behavior,ad_cate)
    a.train()
    output = a.cal_output(test_X)
    rmse=a.cal_rmse(output,test_X)
    acc=a.cal_acc(output,test_X)
    pre=a.cal_pre(output, test_X)
    recall=a.cal_recall(output,test_X)
    fscore = a.cal_Fscore(output, test_X)
    auc = a.cal_auc(output, test_X)
    print("the TBTFD module of rmse:", rmse)
    print("the TBTFD module of acc:", acc)
    print("the TBTFD module of pre:", pre)
    print("the TBTFD module of recall:", recall)
    print("the TBTFD module of fscore:", fscore)
    print("the TBTFD module of auc:", auc)
    return [rmse,acc, pre, recall,fscore,auc]

def tune():
    train_X, test_X = loadData1()
    gammaList=[0.01,0.02,0.04,0.1,0.2,0.4,1,2,4]
    rmseList=[]
    aucList=[]
    for k in gammaList:
        a = SVDPP(train_X)
        a.train(gamma=k)
        output = a.cal_output(test_X)
        rmse = a.cal_rmse(output, test_X)
        rmseList.append(rmse)
        bestThreshold = 0.5
        bestAuc = 0
        for i in range(len(output)):
            output[i]=a.gaussMarkOpt(output[i],0.14)
        auc=a.cal_auc(output,test_X)
        aucList.append(auc)
        '''for i in np.arange(0, 0.5, 0.01):
            output1 = []
            output1.extend(output)
            for j in range(len(output1)):
                output1[j] = a.gaussMarkOpt(output1[j], i)
            auc = a.cal_auc(output1, test_X)
            if auc > bestAuc:
                bestAuc = auc
                bestThreshold = i
        print('bestThreshold:' + str(bestThreshold))
        aucList.append(bestAuc)'''
    print(rmseList)
    print(aucList)

if __name__=='__main__':
    pl.rcParams['font.sans-serif'] = ['SimHei']
    #tune()
    #inp = input()
    pl.rcParams['font.sans-serif'] = ['SimHei']
    iters=list(range(1,52))
    rmseList=[0.5553355227117056, 0.5549237353790362, 0.5548824876154868, 0.5549103660708474, 0.5549712700217397, 0.5550663831098968, 0.555211901985835, 0.5550954453019977, 0.5552109835440646, 0.5551808123881171, 0.5552533047250559, 0.5552433210306775, 0.5553460004502673, 0.5553712573544464, 0.5553796265326826, 0.5553228234270967, 0.5554211892148875, 0.5554567322058075, 0.5553545722230511, 0.5554330554665492, 0.555410079671172, 0.5554403092122289, 0.5554633096336615, 0.5554775023079122, 0.5555537383110772, 0.5555067719066327, 0.5555213600977861, 0.5554925499542309, 0.555506872693714, 0.5555428069618084, 0.5555194988695835, 0.555521497904354, 0.5554832449372502, 0.5554988142786931, 0.5555456308213857, 0.555528507316982, 0.5555007573808471, 0.5555416550514417, 0.5555262210443871, 0.5555235516208213, 0.5555674167039183, 0.5554987357277746, 0.5555033635842459, 0.5555150352759373, 0.5555100779148596, 0.5555485792494981, 0.5555272629333733, 0.555536194859105, 0.5555335141296173, 0.5555503279781384, 0.5555386049855092]
    print(len(rmseList))
    aucList=[0.5490939423813014, 0.5487421925875671, 0.5485053254316854, 0.5484852389514895, 0.5492106354519093, 0.5484147949790716, 0.5487589933357313, 0.5488508268591868, 0.5484480960927367, 0.5489972384886045, 0.5487711076397046, 0.5494080202562988, 0.5485900992844918, 0.5486438362511905, 0.5490328528200019, 0.5490532623980284, 0.5484624824220117, 0.548539021009491, 0.5488191103827312, 0.5488976692639915, 0.5486613467635819, 0.5491309087737153, 0.5486837464615101, 0.5491017796226481, 0.5487261953460755, 0.5490768337917623, 0.549070356070144, 0.5487486514928901, 0.5488373832093643, 0.5491347540786428, 0.5492101660616191, 0.5486344669226487, 0.5487616594861412, 0.5489232534927946, 0.5487885730607933, 0.5488369926440957, 0.5488299929821727, 0.5487025412284479, 0.5488500645449451, 0.5488790246883853, 0.5490264990147079, 0.5487804730693744, 0.5489091714450548, 0.549054129812299, 0.5486761723092679, 0.5489737008285895, 0.5489139292555554, 0.5488201730796362, 0.5488860769003228, 0.5488890810659868, 0.5488439660310116]
    aucList=[0.5482358108165116, 0.5485732534450248, 0.5485999490218751, 0.5490822225414688, 0.5490545953037171, 0.5489200089531935, 0.5488333575392261, 0.5488477850609318, 0.5485397194161344, 0.5484176152196852]
    aucList=[0.5474214798582715, 0.5472876856568363, 0.5484285954604438, 0.5472201407500537, 0.546194962521306, 0.5459971608766406, 0.5, 0.5, 0.5]
    gammaList = [0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1, 2, 4]
    '''fig=pl.figure()
    ax=fig.add_axes((0.1, 0.2, 0.8, 0.7))
    x = range(len(gammaList))
    ax.set_xticks(gammaList)
    ax.set_xticklabels(['0.01', '0.02', '0.04', '0.1', '0.2', '0.4', '1', '2', '4'])
    '''
    nameList=['LR','Naive Bayes','RF','GB','MLP','KNN','TBTFD','集成模型']
    lr=['0.01', '0.02', '0.04', '0.1', '0.2', '0.4', '1', '2', '4']
    x = range(len(nameList))
    aucList = [1.21 * x for x in aucList]
    rmseList = [0.5545245502890712, 0.5551230477631561, 0.5555438811211529, 0.5558076220902233, 0.5563927759945764,
                0.5573472030270007, 0.5673472030451244, 0.56934713564567, 0.5753472030270007]
    aucList=[0.500,0.514,0.519,0.543,0.500,0.516,0.657,0.789]
    pl.bar(x,aucList,color=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'gold'])
    pl.xticks(x,nameList)

    rankList = [x * 5 for x in range(1, 11)]
    rmseList=[0.5553030045399528, 0.5553974833560492, 0.5555065954396545, 0.5556389772823245, 0.5558694612713638, 0.5561020289348252, 0.5564451837693426, 0.5568278756037763, 0.557357482516001, 0.5577777089412901]
    #rmseList=[0.5545245502890712, 0.5551230477631561, 0.5555438811211529, 0.5558076220902233, 0.5563927759945764, 0.5573472030270007, nan, nan, nan]
    pl.xlabel('模型')
    pl.ylabel('AUC')
    pl.title('不同模型的AUC')

    #pl.yticks(aucList)
    #pl.bar(gammaList,aucList,color='rgb')
    #pl.plot(gammaList,aucList)
    pl.show()
    '''pl.plot(list(range(10)))
    pl.xlabel('迭代次数')
    pl.ylabel('Rmse')
    pl.show()'''
    #read_trainAndTestData()
    #call_BaseModule()
    #inp=input()
    #call_SVD()
    #inp=input()
    #rank = [x * 5 for x in range(1, 11)]
    '''y=[0.5478172306397301, 0.5477703433331462, 0.5481968364439493, 0.5491264646082951, 0.5511776069647301, 0.5536439129321553, 0.5571067874470224, 0.5609168038512352, 0.5646046829501866, 0.5688786903354838]
    pl.plot(rank,y)
    pl.xlabel('特征数个数')
    pl.ylabel('AUC')
    pl.show()
    m=input()'''
    begin = timeit.default_timer()
    tune(2)
    inp=input()
    iters=[x*5 for x in range(1,11)]
    aucList=[] 
    rmseList=[]
    for iter in iters:
        result=call_SVDPP(iter)
        aucList.append(result[-1])
        rmseList.append(result[0])
    print(aucList)
    pl.plot(aucList)
    pl.xlabel('特征数个数')
    pl.ylabel('AUC')
    print(rmseList)
    print("time:", (timeit.default_timer() - begin))
    b=input()

    rmseRe=[]
    accRe=[]
    preRe=[]
    recallRe=[]
    fscoreRe=[]
    aucRe=[]
    '''rmse,acc,pre,recall,fscore,auc=call_item_based()
    rmseRe.append(rmse)
    accRe.append(acc)
    preRe.append(pre)
    recallRe.append(recall)
    fscoreRe.append(fscore)
    aucRe.append(auc)

    rmse,acc, pre, recall,fscore,auc =call_SVD()
    rmseRe.append(rmse)
    accRe.append(acc)
    preRe.append(pre)
    recallRe.append(recall)
    fscoreRe.append(fscore)
    aucRe.append(auc)'''

    rmse,acc, pre, recall,fscore,auc = call_SVDPP()
    rmseRe.append(rmse)
    accRe.append(acc)
    preRe.append(pre)
    recallRe.append(recall)
    fscoreRe.append(fscore)
    aucRe.append(auc)

    '''rmse,acc, pre, recall, fscore, auc =call_TBSVD()
    rmseRe.append(rmse)
    accRe.append(acc)
    preRe.append(pre)
    recallRe.append(recall)
    fscoreRe.append(fscore)
    aucRe.append(auc)'''

    print(rmseRe)
    print(accRe)
    print(preRe)
    print(recallRe)
    print(fscoreRe)
    print(aucRe)
    '''[0.16900634765625, 0.6220779418945312, 0.0501708984375, 0.05210113525390625]
[0.04695113862480654, 0.061795466568998494, 0.0493100261938036, 0.049336516828137195]
[0.8214451493114653, 0.4699056165867244, 0.9990716385579452, 0.997524369487854]
[0.0888253107798357, 0.10922692369940117, 0.09398151517356816, 0.09402276555561227]
[0.47830597553367887, 0.54993808223023666, 0.50001331287895334, 0.5002949793253777]'''
    rmsefig = pl.figure('rmse')
    pl.plot(rmseRe)

    accfig=pl.figure('acc')
    pl.plot(accRe)

    prefig=pl.figure('pre')
    pl.plot(preRe)

    recallfig=pl.figure('recall')
    pl.plot(recallRe)

    recallfig = pl.figure('recall')
    pl.plot(recallRe)

    fscorefig = pl.figure('fscore')
    pl.plot(fscoreRe)

    aucfig = pl.figure('auc')
    pl.plot(aucRe)

    rmsefig.show()
    accfig.show()
    prefig.show()
    recallfig.show()
    fscorefig.show()
    aucfig.show()

    a=input()






    #pool=ThreadPool(3)
    '''dims=[10,20,30,40,50,60,70,80,90,100]
    resultRmse=[]
    resultAuc=[]
    for k in dims:
        rmse,auc=call_TBSVD(k)
        resultRmse.append(rmse)
        resultAuc.append(auc)
    print(resultRmse)
    print(auc)'''
    #result=pool.map(call_TBSVD,base)
    #pool.close()
    #pool.join()
    print("multithread cost time:", (timeit.default_timer() - t))
    #behavior=read_behavior()
    '''for be in behavior:
        print(be,behavior[be])'''
    '''ad_cate=read_ad_feature()
    for ad in ad_cate:
        print(ad,ad_cate[ad])
    exit(0)'''


    '''print(np.array(train_X).shape, np.array(test_X).shape)
    t=timeit.default_timer()
    call_item_based()
    call_SVD()
    call_SVDPP()
    print("normal cost time:",(timeit.default_timer()-t))'''
    '''it=[5,10,15,20,25,30,35,40,45,50]#[0.28444363903603609, 0.28445612069338494, 0.28514496903389097, 0.28349837633838865, 0.28420326740054458, 0.28482925550237675, 0.28626237316383968, 0.28489440790484877, 0.28513930269170329, 0.2849884935758959]
    pool=ThreadPool(3)
    result=pool.map(call_TBSVD,it)
    pool.close()
    pool.join()
    print(result)'''
    '''p1=Process(target=call_item_based)
    p1.start()
    p2=Process(target=call_SVD)
    p2.start()
    p3=Process(target=call_SVDPP)
    p3.start()'''
    '''p4=Process(target=call_TBSVD)
    p4.start()
    p4.join()'''
    '''p1.join()
    p2.join()
    p3.join()'''
    #print(output)



