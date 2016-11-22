import numpy as np
import scipy.optimize as spo
import scipy.stats as sps
from myquant.derivative import *
import time

class Arima():
    
    def __init__(self, x, ar = 1, di = 0, ma = 1, mid = True, seed = 1000):
        self.x = x
        self.ar = ar
        self.di = di
        self.ma = ma
        rng = np.random.RandomState(seed)
        self.pars = rng.uniform(0,1,ar+ma+1)
        self.pars[-1] = 0
        self.mid = mid
    
    def get_pars(self,pars):
        loc = 0
        ars = pars[loc:(loc+self.ar)]
        loc += self.ar
        mas = pars[loc:(loc+self.ma)]
        loc += self.ma
        mu = pars[-1]
        return ars,mas,mu
        
    def re_diff(self,data,keys):
        res = data * 1
        for key in keys[::-1]:
            res = np.concatenate((np.array(key*1).reshape(1),res))
            for i in range(1,len(res)):
                res[i] += res[i-1]
        return res
                        
    def diff(self,data,di = 1):
        res = data * 1
        keys = []
        for i in range(di):
            keys.append(res[0])
            res = res[1:] - res[:-1]
        return res,keys
        
    def init_series(self):
        max_loc = np.max([self.ar,self.ma])
        self.start_loc = max_loc
        x,self.keys = self.diff(self.x,self.di)
        self.x_series = np.concatenate((np.repeat(0,max_loc),x))
        self.hlen = len(self.x_series)
    
    def get_loss(self,pars,renew_params = False):
        
        #load params
        x = self.x_series
        ars,mas,mu = self.get_pars(pars)
        if self.mid == False:
            mu = 0
        hlen = self.hlen
        
        #get ep
        ep = np.zeros(hlen)
        rp = np.zeros(hlen)
        if self.ar == 0 and self.ma == 0:
            ep = x - mu
        else:
            for i in range(self.start_loc,hlen):
                rp[i] = mu + x[(i-self.ar):i].dot(ars) + ep[(i-self.ma):i].dot(mas)
                ep[i] = x[i] - rp[i]
        
        N = hlen - self.start_loc
        Ed = np.sum(ep**2)
        sigma2 = Ed/N
        #likelihood
        L = - N/2*np.log(2*np.pi) - N/2*np.log(sigma2) - 1/sigma2*Ed/2
        
        if renew_params:
            self.rps = rp[self.start_loc:]
            
        return -L
        
    def train(self):
        self.init_series()
        start = time.clock()
        temp = spo.minimize(self.get_loss, self.pars, method = 'L-BFGS-B')
        end = time.clock()
        print(str(end-start)+' secs used')
        if not temp['success']:
            print('Warn: optimization failed')
        self.pars = temp['x']
        self.get_loss(self.pars,True)
        return temp['x']
        
    def get_next(self,ahead = 5):
        #load params
        x = np.concatenate((self.x_series,np.repeat(0,ahead)))
        ars,mas,mu = self.get_pars(self.pars)
        if self.mid == False:
            mu = 0
        hlen = self.hlen
        
        #get ep
        ep = np.zeros(hlen+ahead)
        rp = np.zeros(hlen+ahead)
        if self.ar == 0 and self.ma == 0:
            ep = x - mu
        else:
            for i in range(self.start_loc,hlen+ahead):
                rp[i] = mu + x[(i-self.ar):i].dot(ars) + ep[(i-self.ma):i].dot(mas)
                if i < hlen:
                    ep[i] = x[i] - rp[i]
        
        res_rp = rp[hlen:]
        
        if self.di > 0:
            keys = [self.x[-1] * 1]
            for i in range(self.di - 1):
                keys.append(self.diff(self.x,i)[0][-1])
            keys = keys[::-1]
            for key in keys:
                res_rp = self.re_diff(res_rp,[key])[-ahead:]
                
        return res_rp
        
    def get_param_stats(self,ld = 0.0001,af = 0.0001):
        
        hlen = len(self.pars)
        H = get_hessian(self.get_loss,self.pars,ld)
        try:
            Hi = np.linalg.inv(H)
        except:
            Hi = np.linalg.inv(H+np.eye(hlen)*af)
        sigma = np.array([np.sqrt(np.abs(Hi[i,i])) for i in range(hlen)])
        if np.isnan(self.pars/sigma).any():
            print('Warn: NaN Produced')
        t_values = np.nan_to_num(self.pars/sigma)
        p_values = 2*(1-sps.t(len(self.x)-len(self.pars)).cdf(np.abs(t_values)))
        
        return sigma,p_values
        
if __name__ == '__main__':
    
    import tushare as ts
    data = ts.get_h_data('000001',index = True,start = '2010-01-01',end = '2016-05-06')
    data = data.sort()
    price = data['close'].values
    ret = price[1:]/price[:-1] - 1
    
    arima = Arima(ret[-51:-1],2,0,0)
    res = arima.train()
    sigma,p_values = arima.get_param_stats()
    print(np.round(p_values,3))
    
    from myquant.linear_method import *
    from myquant.functions_z1 import *
    
    temp = before(np.concatenate((np.repeat(0,2),ret[-51:-1])),3)
    linear(temp[:,:2],temp[:,2])
    
    record = []
    for i in range(100):
        arima = Arima(ret[(i-150):(i-100)],1,0,1)
        res = arima.train()
        rp = arima.get_next(1)
        record.append(rp[0])
        print(i)
    record = np.array(record)
    
    temp = ret[-100:]
    print(np.sum(temp>0)/len(temp))
    print(np.sum(temp[(p<0.7)&(p>0.5)]>0)/len(temp[(p<0.7)&(p>0.5)]))
    print(np.sum(temp[record>0]<0)/len(temp[record>0]))
    
    
    
    
    
    
    
    