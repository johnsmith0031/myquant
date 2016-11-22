import numpy as np
from myquant.linear_method import *
import time

class GB():
    
    def __init__(self,x,y,sigma2_w):
        self.x = x
        self.y = y
        self.hlen = len(x)
        self.sigma2_e = np.var(linear(self.x,self.y)[3])
        self.sigma2_w = sigma2_w
        
    def init_series(self):
        b0 = linear(self.x,self.y)[0][1]
        self.betas = np.ones(self.hlen+1) * b0
        self.F = 1
    
    def get_sigma2_w(self):
        res = np.var(self.betas[2:] - self.F * self.betas[1:-1])
        return res
        
    def get_sigma2_e(self):
        res = np.var(self.y - self.betas[1:]*self.x)
        return res
    
    def p_bi(self,betap,i):
        x,y = self.x,self.y
        F = self.F
        sigma2_w,sigma2_e = self.sigma2_w,self.sigma2_e
        betas = self.betas
        if i == 0:
            res = np.exp(-((beta[i+1]-F*betap)**2)/(2*sigma2_w))
        elif i < self.hlen:
            res = np.exp(-(y[i-1]-betap*x[i-1])**2/(2*sigma2_e)-
                          ((betap-F*betas[i-1])**2+(betas[i+1]-F*betap)**2)/(2*sigma2_w)-
                          (betap-betas[i-1])**2/(2*sigma2_w))
        else:
            res = np.exp(-(y[i-1]-betap*x[i-1])**2/(2*sigma2_e)-
                          (betap-F*betas[i-1])**2/(2*sigma2_w)-
                          (betap-betas[i-1])**2/(2*sigma2_w))
        return res
        
    def sample_bi(self,i,flag = False):
        if flag:
            tx = np.random.randn(10)
            for j in range(1,10):
                r = np.min([self.p_bi(tx[j],i)/self.p_bi(tx[j-1],i),1])
                if np.random.rand() > r:
                    tx[j] = tx[j-1] * 1
            res = tx[-1]
        else:
            x,y = self.x,self.y
            F = self.F
            sigma2_w,sigma2_e = self.sigma2_w,self.sigma2_e
            betas = self.betas
            if i == 0:
                res = betas[i+1] / F
            elif i < self.hlen:
                res = (F*sigma2_e*(betas[i-1]+betas[i+1])+sigma2_w*x[i-1]*y[i-1])/\
                      (sigma2_e+F**2*sigma2_e+sigma2_w*x[i-1]**2)
            else:
                res = (F*sigma2_e*betas[i-1]+sigma2_w*x[i-1]*y[i-1])/\
                      (sigma2_e+sigma2_w*x[i-1]**2)
        return res
        
    def sample_F(self,flag = False):
        X = self.betas[1:-1].reshape((-1,1))
        Y = self.betas[2:].reshape((-1,1))
        Fu = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        if flag:
            Fv = np.sqrt(self.sigma2_w*np.linalg.inv(X.T.dot(X)))
            res = np.random.rand()*Fv + Fu
        else:
            res = 1
        return res

    def run(self,flag = False):
        for i in range(self.hlen+1):
            self.betas[i] = self.sample_bi(i)
        if flag:
            self.sigma2_e = self.get_sigma2_e()
            self.sigma2_w = self.get_sigma2_w()

    def train(self,max_iter = 1000,tol = 1e-6,flag = False):
        start = time.clock()
        self.init_series()
        record = np.array([]).reshape((0,self.hlen))
        for i in range(max_iter):
            self.run(flag)
            record = np.row_stack((record,self.betas[1:]))
            if i > 2:
                temp = np.sum((record[-1,:] - record[-2,:])**2)
                if temp < tol:
                    break
        end = time.clock()
        print(str(end-start)+' secs used')

if __name__ == '__main__':
    
    import tushare as ts
    
    data = ts.get_h_data(code = '000001', index = True, start = '2014-01-01',end = '2016-05-06')
    data = data.sort()
    data2 = ts.get_h_data(code = '000001', start = '2014-01-01',end = '2016-05-06')
    data2 = data2.sort()
    data2 = data2.reindex(data.index)
    data2 = data2.fillna(method = 'ffill')
    price = data['close'].values * 1
    ret1 = price[1:]/price[:-1] - 1
    price = data2['close'].values * 1
    ret2 = price[1:]/price[:-1] - 1
    dates = np.array(data.index,dtype='<U10')
    
    
    x = ret1[:221]
    y = ret2[:221]
    gb = GB(x,y,0.1)
    gb.train(500,1e-6)
    plt.plot(gb.betas[1:],'r-')
    
    bts = gb.betas[1:]
    res1 = y - bts*x
    res2 = linear(x,y)[3]
    print(np.std(res1))
    print(np.std(res2))


    linear(ret1[:100],ret2[:100])[0][1]











