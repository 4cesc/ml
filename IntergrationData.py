def timeToInt(time):
    if time>=1494000000 and time<1494086400:
        return 1
    if time>=1494086400 and time<1494172800:
        return 2
    if time>=1494172800 and time<1494259200:
        return 3
    if time>=1494259200 and time<1494345600:
        return 4
    if time>=1494345600 and time<1494432000:
        return 5
    if time>=1494432000 and time<1494518400:
        return 6
    if time>=1494518400 and time<1494604800:
        return 7
    if time>=1494604800 and time<1494691200:
        return 8
    return 0

def convertBtag(s):
    if s=='pv':
        return 1
    if s=='cart':
        return 2
    if s=='fav':
        return 3
    if s=='buy':
        return 4
    return 0

def intergData():
    ad_feature = {}
    with open('ad_feature.csv') as f1:
        f1.readline()
        afData=f1.readlines()
        ad_feature={}
        for line in afData:
            #line = line.replace(",,", ",NULL,")
            data=line.strip('\n').split(',',1)
            #print(len(data[1].split(',')))
            ad_feature[int(data[0])]=data[1]

    user_profile = {}
    with open('user_profile.csv') as f2:
        f2.readline()
        upData = f2.readlines()
        for line in upData:
            line=line.replace(",,",",NULL,")
            data = line.strip('\n').split(',', 1)
            if data[1][-1]==',':
                data[1]+="NULL"
            user_profile[int(data[0])] = data[1]

    f=open('data.csv','w+')
    f.write('uid,adid,cms_segid,cms_group_id,final_gender_code,age_level,pvalue_level,shopping_level,occupation,new_user_class_level,cate_id,campaign_id,customer,brand,price,time,pid,clk\n')
    with open('raw_sample.csv') as f3:
        rsData=f3.readlines()
        for line in rsData:
            data=line.split(',')
            uid,adid,time,pid,clk=int(data[0]),int(data[2]),data[1],data[3],data[5]
            userData=user_profile[uid] if uid in user_profile else "NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL"
            adData=ad_feature[adid] if adid in ad_feature else "NULL,NULL,NULL,NULL,NULL"
            s=""
            s+=str(uid)
            s+=","
            s+=str(adid)
            s+=","
            s+=userData
            s+=","
            s+=adData
            s+=","
            s+=str(timeToInt(int(time)))
            s+=","
            s+=str(pid)
            s+=","
            s+=str(clk)#clk后面刚好有个\n
            f.write(s)
    f.close()
def interGrationBehavior():
    rs={}
    with open('raw_sample.csv') as f1:
        f1.readline()
        rsData=f1.readlines()
        for line in rsData:
            #line = line.replace(",,", ",NULL,")
            data=line.split(',',1)
            #print(len(data[1].split(',')))
            rs[data[0]]=data[1]

    f=open('filterPvBehavior.csv','w+')
    f.write('uid,time,btag,cate,brand,adid,timeNow,pid,clk\n')
    with open('behavior_log','r',encoding='latin1') as f2:
        line=f2.readline()
        while line:
            line=f2.readline()
            data=line.strip('\n').split(',')
            if data[0] not in rs:
                continue
            uid,time,btag,cate,brand=data[0],data[1],convertBtag(data[2]),data[3],data[4]
            if btag==1:
                continue
            rawData=rs[data[0]].split(',')
            adid,timeNow,pid,clk=rawData[0],timeToInt(int(rawData[1])),rawData[2],rawData[4]
            s=''
            s+=uid+','+time+','+str(btag)+','+cate+','+brand+','+adid+','+str(timeNow)+','+pid+','+clk
            f.write(s)
    f.close()

if __name__=='__main__':
    #intergData()
    interGrationBehavior()

