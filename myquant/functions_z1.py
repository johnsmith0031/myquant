import numpy as np

def is_in(ele,tlist):
    for tele in tlist:
        if ele == tele:
            return True
    return False

def to_one(data):
    n, p = data.shape
    MaxMin = []
    res = np.zeros(data.shape)
    for i in range(p):
        M = np.max(data[:,i])
        m = np.min(data[:,i])
        res[:,i] = (data[:,i]-m)/(M-m)
        MaxMin.append((M,m))
    return (res,MaxMin)
    
def de_one(data,MaxMin):
    n, p = data.shape
    res = np.zeros(data.shape)
    for i in range(p):
        M, m = MaxMin[i]
        res[:,i] = data[:,i]*(M-m) + m
    return res
    
def de_outlier(data,a=5):
    upper = np.percentile(data,100-a,axis=0)
    lower = np.percentile(data,a,axis=0)
    loc1 = (data <= upper)
    loc2 = (data >= lower)
    loc = np.array(np.prod(loc1*loc2,axis=1),dtype=bool)
    res = data[loc,:]
    return (res,loc)
    
def adcode(codes):
    res = []
    for code in codes:
        for j in range(6 - len(code)):
            code = '0' + code
        res.append(code)
    return res

def retraction(data):
    hlen = len(data)
    Max = data[0]
    res = 0
    for i in range(1,hlen):
        if data[i] > Max:
            Max = data[i]
        else:
            res = np.max([(Max - data[i])/Max,res])
    return res

def before(data,n):
    hlen = len(data)
    record = []
    for i in range(n,hlen+1):
        record.append(data[(i-n):i])
    return np.array(record)