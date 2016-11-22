import numpy as np
from myquant.linear_method import *

class DBT():
    
    def __init__(self,price):
        self.price = price
        
    def DB(self):
        data = self.price
        hlen = len(data)
        Max = data[0]
        res = 0
        days = 0
        record = []
        days_record = []
        res_record = []
        for i in range(1,hlen):
            if data[i] > Max:
                Max = data[i]
            res = (Max - data[i])/Max
            if res == 0 and len(record) == 0:
                continue
            record.append(res)
            if res > 0 and i < hlen - 1:
                days += 1
            else:
                days_record.append(days*1)
                res_record.append(np.max(record))
                record = []
                days = 0
        return np.array(days_record),np.array(res_record)

def RTR(data):
    dbt = DBT(data)
    a,b = dbt.DB()
    loc = np.argsort(b)[-5:]
    res = RAR(data)/((np.mean(a[loc])/250)*np.mean(b[loc]))
    return res
    
def MDB(data):
    hlen = len(data)
    Max = data[0]
    res = []
    for i in range(1,hlen):
        if data[i] > Max:
            Max = data[i]
        res.append((Max - data[i])/Max)
    return np.max(res)

def ANR(data):
    res = ((data[-1]-data[0])/data[0]/len(data))*250
    return res

def CAGR(data):
    hlen = len(data)
    res = (data[-1]/data[0])**(250/hlen)-1
    return res

def MAR(data):
    mdb = MDB(data)
    anr = ANR(data)
    res = anr/mdb
    return res
    
def Sharpe(data, rf = 0.035):
    ret = data[1:]/data[:-1] - 1
    res = (np.mean(ret)*250 - rf)/np.sqrt(250)/np.std(ret)
    return res

def RAR(data):
    pt = data * 1
    hlen = len(pt)
    X = np.array(range(hlen))/hlen
    beta,sigma,R2,res = linear(X,pt)
    res = beta[1]/pt[0]/hlen*250
    return res

def CRAR(data):
    pt = np.log(data/data[0]*100)
    hlen = len(pt)
    X = np.array(range(hlen))/hlen
    beta,sigma,R2,res = linear(X,pt)
    res = beta[1]*(250/hlen)
    return res
    
if __name__ == '__main__':
    
    print('ok')
    
    
    
    
    