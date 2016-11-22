import numpy as np
import math

class Hurst():
    
    def __init__(self,series,max_num = None):
        self.series = series
        if max_num is not None:
            cum_s = 0
            res = np.ones(max_num)
            for n in range(2,max_num):
                i = np.array(range(1,n))
                cum_s = np.sum(np.sqrt((n-i)/i))
                if n <= 340:
                    res[n] = math.gamma((n-1)/2)/(math.gamma(n/2)*np.sqrt(math.pi)) * cum_s
                else:
                    res[n] = 1/np.sqrt(n*math.pi/2) * cum_s
            self.adj = res
    
    def R_S(self,a,b):
        series = self.series[a:b]
        mu = np.mean(series)
        Y = series-mu
        Z = np.cumsum(Y)
        R = np.max(Z) - np.min(Z)
        S = np.sqrt(np.mean(Y**2))
        if S == 0:
            res = 1
        else:
            res = R/S
        return res
    
    def hurst(self,a,b):
        hurst = np.log(self.R_S(a,b))/np.log(b-a)
        return hurst
    
    def hurst_old(self,a,b,ld = 0.0001):
        res = []
        for i in range(a+1,b):
            res.append([i-a,self.R_S(a,i)])
        res = np.array(res)
        X = np.column_stack((np.repeat(1,len(res)),np.log(res[:,0])))
        Y = np.log(res[:,1])
        temp = X.T.dot(X)
        try:
            beta = np.linalg.inv(temp).dot(X.T).dot(Y)
        except:
            beta = np.linalg.inv(temp + np.eye(len(temp))*ld).dot(X.T).dot(Y)
        hurst = beta[1]
        return hurst
    
    def hurst_adjusted(self,a,b,ld = 0.0001):
        res = []
        for i in range(a+1,b):
            res.append([i-a,self.R_S(a,i),self.adj[i-a]])
        res = np.array(res)
        res = np.log(res)
        X = np.column_stack((np.repeat(1,len(res)),res[:,0]))
        #X = res[:,0].reshape((-1,1))
        Y = (res[:,1] - res[:,2]).reshape((-1,1))
        temp = X.T.dot(X)
        try:
            beta = np.linalg.inv(temp).dot(X.T).dot(Y)
        except:
            beta = np.linalg.inv(temp + np.eye(len(temp))*ld).dot(X.T).dot(Y)
        hurst = beta[-1,0] + 0.5
        return hurst
        
    def confidence(self,N):
        M = np.log(N)/np.log(2)
        lower = 0.5 - np.exp(-7.33*np.log(M) + 4.21)
        upper = np.exp(-7.20*np.log(M) + 4.04) + 0.5
        #lower = 0.5 - np.exp(-2.93*np.log(M) + 4.45)
        #upper = np.exp(-3.10*np.log(M) + 4.77) + 0.5
        return lower,upper
    
if __name__ == '__main__':
    
    series = np.random.randn(1000)
    h = Hurst(series)
    series = np.random.randn(1000)
    h.series = series
    h.hurst(0,1000)
    
    series = np.random.randn(1000)
    h = Hurst(series)
    res = []
    for i in range(1000):
        series = np.random.randn(50)
        h.series = series
        res.append(h.hurst(0,len(series)))
        print(i)
    res = np.array(res)
    upper = np.mean(res) + np.std(res)*1.64
    lower = np.mean(res) - np.std(res)*1.64

