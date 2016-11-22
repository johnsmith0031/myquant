import numpy as np
import scipy.optimize as spo
import scipy.stats as sps
from derivative import *
from linear_method import *
import time

class Garch_Xreg():
    
    def __init__(self, x, xreg, ar = 1, ma = 1, beta = 1, omega = 1, s1 = 1, s2 = 1, mid = True, seed = 1000):
        self.x = x
        self.xreg = xreg
        self.ar = ar
        self.ma = ma
        self.beta = beta
        self.omega = omega
        self.s1 = s1
        self.s2 = s2
        rng = np.random.RandomState(seed)
        self.pars = rng.uniform(0,1,ar+ma+beta+omega+s1+s2+3)
        self.pars[ar+ma] = 0.8
        self.pars[ar+ma+beta] = 0.1
        self.pars[ar+ma+beta+omega] = 0.8
        self.pars[-2] = 0
        self.bt0 = linear(xreg,x)[0][1]
        self.var0 = np.std(x)**2
        self.mid = mid
        self.pars[-3] = self.bt0 * 1
        self.pars[-1] = self.var0 * 1
        
    def get_pars(self,pars):
        
        loc = 0
        ars = pars[loc:(loc+self.ar)]
        loc += self.ar
        mas = pars[loc:(loc+self.ma)]
        loc += self.ma
        betas = pars[loc:(loc+self.beta)]
        loc += self.beta
        omegas = pars[loc:(loc+self.omega)]
        loc += self.omega
        s1s = pars[loc:(loc+self.s1)]
        loc += self.s1
        s2s = pars[loc:(loc+self.s2)]
        loc += self.s2
        uxreg = pars[-3]
        ur = pars[-2]
        uo = pars[-1]
        
        return ars,mas,betas,omegas,s1s,s2s,uxreg,ur,uo
    
    def init_series(self):
        max_loc = np.max([self.ar,self.ma,self.beta,self.omega,self.s1,self.s2])
        self.start_loc = max_loc
        self.x_series = np.concatenate((np.repeat(0,max_loc),self.x))
        self.xreg_series = np.concatenate((np.repeat(0,max_loc),self.xreg))
        self.hlen = len(self.x_series)
    
    def get_loss(self,pars,renew_params = False):
        
        #load params
        x = self.x_series
        xreg = self.xreg_series
        ars,mas,betas,omegas,s1s,s2s,uxreg,ur,uo = self.get_pars(pars)
        if self.mid == False:
            uxreg,ur,uo = 0,0,0
        hlen = self.hlen
        
        #get ep
        ep = np.zeros(hlen)
        rp = np.zeros(hlen)
        bt = np.ones(hlen) * self.bt0
        if self.ar == 0 and self.ma == 0 and self.s1 == 0 and self.s2 == 0:
            bt[:] = uxreg
            ep = x - ur - bt * xreg
        else:
            for i in range(self.start_loc,hlen):
                bt[i] = uxreg + bt[(i-self.s1):i].dot(s1s) + np.abs(ep[(i-self.s2):i]).dot(s2s)
                rp[i] = ur + x[(i-self.ar):i].dot(ars) + ep[(i-self.ma):i].dot(mas) + bt[i] * xreg[i]
                ep[i] = x[i] - rp[i]

        #get op
        ep2 = ep**2
        op = np.ones(hlen) * self.var0
        if self.beta == 0 and self.omega == 0:
            op[:] = uo
        else:
            for i in range(self.start_loc,hlen):
                op[i] = uo + op[(i-self.beta):i].dot(betas) + ep2[(i-self.omega):i].dot(omegas)
        
        #likelihood
        L = np.sum(np.log(op[self.start_loc:])/2 + ep2[self.start_loc:]/(2*op[self.start_loc:]))
        
        if renew_params:
            self.bts = bt[self.start_loc:]
            self.sigma = np.sqrt(op[self.start_loc:])
            self.rps = rp[self.start_loc:]
            
        return L
        
    def train(self,bounds = True):
        self.init_series()
        bnds = None
        if bounds:
            bnds = []
            loc = 0
            for i in range(loc,loc+self.ar+self.ma):
                bnds.append([-np.Inf,np.Inf])
            loc += self.ar+self.ma
            for i in range(loc,loc+self.beta+self.omega):
                bnds.append([0,np.Inf])
            loc += self.beta+self.omega
            for i in range(loc,loc+self.s1+self.s2):
                bnds.append([-np.Inf,np.Inf])
            bnds.append([-np.Inf,np.Inf])
            bnds.append([-np.Inf,np.Inf])
            bnds.append([0,np.Inf])
        start = time.clock()
        temp = spo.minimize(self.get_loss,self.pars , bounds = bnds, method = 'L-BFGS-B')
        end = time.clock()
        print(str(end-start)+' secs used')
        if not temp['success']:
            print('Warn: optimization failed')
            
        self.pars = temp['x']
        self.get_loss(self.pars,True)
        return temp['x']
        
    def get_next(self,ahead = 5):
        x = np.concatenate((self.x_series,np.repeat(0,ahead)))
        xreg = np.concatenate((self.xreg_series,np.repeat(0,ahead)))
        ars,mas,betas,omegas,s1s,s2s,uxreg,ur,uo = self.get_pars(self.pars)
        if self.mid == False:
            uxreg,ur,uo = 0,0,0
        hlen = self.hlen
        
        #get ep
        ep = np.zeros(hlen + ahead)
        rp = np.zeros(hlen + ahead)
        bt = np.ones(hlen + ahead) * self.bt0
        if self.ar == 0 and self.ma == 0 and self.s1 == 0 and self.s2 == 0:
            bt[:] = uxreg
            ep = x - ur - bt * xreg
        else:
            for i in range(self.start_loc,hlen + ahead):
                bt[i] = uxreg + bt[(i-self.s1):i].dot(s1s) + np.abs(ep[(i-self.s2):i]).dot(s2s)
                rp[i] = ur + x[(i-self.ar):i].dot(ars) + ep[(i-self.ma):i].dot(mas) + bt[i] * xreg[i]
                ep[i] = x[i] - rp[i]

        #get op
        ep2 = ep**2
        op = np.ones(hlen + ahead) * self.var0
        if self.beta == 0 and self.omega == 0:
            op[:] = uo
        else:
            for i in range(self.start_loc,hlen + ahead):
                op[i] = uo + op[(i-self.beta):i].dot(betas) + ep2[(i-self.omega):i].dot(omegas)
        
        res_rp = rp[hlen:]
        res_op = np.sqrt(op[hlen:])
        res_bt = bt[hlen:]
        
        return res_rp,res_op,res_bt
        
    def get_param_stats(self,ld = 0.0001,af = 0.0001):
        
        hlen = len(self.pars)
        #J = get_jacobian(self.get_loss,self.pars,ld)
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
    
    garch = Garch_Xreg(ret2[-100:],ret1[-100:],0,0,1,1,1,1)
    res = garch.train()
    rp,op,bt = garch.get_next()
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(garch.bts)
    ax2.plot(garch.sigma,'g-')
    
    linear(ret1,ret2)[0][1]
    
    
    
    garch = Garch(ret[-50:],0,0,1,1)
    res = garch.train()
    temp = garch.sigma
    plt.plot(temp)
    rp,op = garch.get_next(1)
    sigma,p_values = garch.get_param_stats(ld = 1e-4,af = 1e-4)
    print(np.round(p_values,4))
    
    temp = np.random.randn(100)
    garch = Garch(temp,0,0,1,1)
    res = garch.train()
    plt.plot(garch.sigma)
    H,sigma,p_values = garch.get_param_stats(1e-8)
    print(p_values)
        
    import tushare as ts
    
    data = ts.get_h_data(code = '000001', index = True, start = '2014-01-01',end = '2016-05-06')
    data = data.sort()
    data2 = ts.get_h_data(code = '601886', start = '2014-01-01',end = '2016-05-06')
    data2 = data2.sort()
    data2 = data2.reindex(data.index)
    data2 = data2.fillna(method = 'ffill')
    price = data['close'].values * 1
    ret1 = price[1:]/price[:-1] - 1
    price = data2['close'].values * 1
    ret2 = price[1:]/price[:-1] - 1
    
    import matplotlib.pyplot as plt
    
    garch = Garch(ret[-50:],0,0,1,1,True)
    res = garch.train(cons_flag = False,bounds = True)
    temp = garch.sigma
    rp,op = garch.get_next()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(temp,'b-')
    ax2.plot(price[-50:],'g-')
    
    rps = garch.rps
    sigma = garch.sigma
    hlen = len(rps)
    p = [1-sps.norm(rps[i],sigma[i]).cdf(0) for i in range(hlen)]
    p = np.array(p)
    
    temp = ret[-50:]
    temp = temp[p>0.5]
    np.sum(temp>0)/len(temp)
    
    temp = ret[-50:]
    temp = temp[p<0.5]
    np.sum(temp>0)/len(temp)
    
    record = []
    for i in range(500):
        garch = Garch(ret[(i-1050):(i-1000)],1,1,1,1)
        res = garch.train()
        temp = garch.sigma
        rp,op = garch.get_next(1)
        record.append([rp[0],op[0]])
        print(i)
    record = np.array(record)
    
    p = []
    for i in range(len(record)):
        p.append(1-sps.norm(record[i,0],record[i,1]).cdf(0))
    p = np.array(p)
    
    bars = []
    for i in range(len(record)):
        bars.append([sps.norm(record[i,0],record[i,1]).ppf(0.025),
                     sps.norm(record[i,0],record[i,1]).ppf(0.975)])
    bars = np.array(bars)
    
    temp = ret[-1000:-500]
    
    loc = (temp < bars[:,0]) | (temp > bars[:,1])
    np.sum(loc)/500
    
    
    plot(temp,'g-')
    plot(bars[:,0],'b-')
    plot(bars[:,1],'b-')
    
    np.sum(temp>0)/len(temp)
    np.sum(temp[p>0.5]>0)/len(temp[p>0.5])
    np.sum(temp[p<0.5]<0)/len(temp[p<0.5])
    
    garch = Garch(ret[-50:],0,0,1,1)
    res = garch.train()
    sigma_t = garch.sigma
    
    sigma_p = record[:,1]
    
    temp = np.column_stack((sigma_p,sigma_t))
    plt.plot(temp[:,0],'b-')
    plt.plot(temp[:,1],'g-')
    
    ema = [record[0,0]*1]
    for i in range(1,500):
        ema.append(ema[-1]*0.9 + record[i,0]*0.1)
    ema = np.array(ema)
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(ema,'b-')
    ax1.plot(np.repeat(0,500),'r-')
    ax2.plot(price[-1000:-500],'g-')
    
    
    
    
    
    
    
    
    
    