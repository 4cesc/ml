import yagmail
def sendEmail():
    #yagmail.register('user','password')lqszkpalqsvcdiic
    yag=yagmail.SMTP(user='2449987483@qq.com',password='lqszkpalqsvcdiic',host='smtp.qq.com',port='465')
    content='测试内容'
    att=['C:/Users/yangfang07/Desktop/论文/feature_importance.png','C:/Users/yangfang07/Desktop/论文/report.pdf']
    yag.send('2449987483@qq.com','test',contents=content,attachments=att)
    pass

if __name__=='__main__':
    sendEmail()