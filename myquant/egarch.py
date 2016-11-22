import numpy as np
import scipy.optimize as spo
import scipy.stats as sps
from myquant.derivative import *
import time

class EGarch():
    
    def __init__(self, x, ar = 0, ma = 0, beta = 1, omega = 1, mid = True,seed = 1000):
        self.x = x
        self.ar = ar
        self.ma = ma
        self.beta = beta
        self.omega = omega
        rng = np.random.RandomState(seed)
        self.pars = rng.uniform(0,1,ar+ma+beta+omega+omega+2)
        self.pars[ar+ma] = 0.8
        self.pars[ar+ma+beta] = 0.1
        self.pars[-2] = 0
        self.pars[-1] = 0
        self.var0 = np.std(x)**2
        self.mid = mid
    
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
        lnys = pars[loc:(loc+self.omega)]
        ur = pars[-2]
        uo = pars[-1]
        
        return ars,mas,betas,omegas,lnys,ur,uo
    
    def init_series(self):
        max_loc = np.max([self.ar,self.ma,self.beta,self.omega])
        self.start_loc = max_loc
        self.x_series = np.concatenate((np.repeat(0,max_loc),self.x))
        self.hlen = len(self.x_series)
    
    def get_loss(self,pars,renew_params = False):
        
        #load params
        x = self.x_series
        ars,mas,betas,omegas,lnys,ur,uo = self.get_pars(pars)
        if self.mid == False:
            ur,uo = 0,0
        hlen = self.hlen
        
        #get ep
        ep = np.zeros(hlen)
        rp = np.zeros(hlen)
        if self.ar == 0 and self.ma == 0:
            ep = x - ur
        else:
            for i in range(self.start_loc,hlen):
                rp[i] = ur + x[(i-self.ar):i].dot(ars) + ep[(i-self.ma):i].dot(mas)
                ep[i] = x[i] - rp[i]

        #get op
        ep2 = ep**2
        op = np.ones(hlen) * self.var0
        if self.beta == 0 and self.omega == 0:
            op = uo
        else:
            for i in range(self.start_loc,hlen):
                tp = (np.abs(ep[(i-self.omega):i])+lnys*ep[(i-self.omega):i])/\
                     np.sqrt(op[(i-self.beta):i])
                op[i] = np.exp(uo + np.log(op[(i-self.beta):i]).dot(betas) + tp.dot(omegas))
                        
        
        #likelihood
        L = np.sum(np.log(op[self.start_loc:])/2 + ep2[self.start_loc:]/(2*op[self.start_loc:]))
        
        if renew_params:
            self.sigma = np.sqrt(op[self.start_loc:])
            self.rps = rp
            
        return L
        
    def train(self,bounds = True):
        self.init_series()
        bnds = None
        if bounds:
            bnds = []
            for i in range(0,self.ar+self.ma):
                bnds.append([-np.Inf,np.Inf])
            for i in range(self.ar+self.ma,self.ar+self.ma+self.omega+self.beta):
                bnds.append([-np.Inf,np.Inf])
            for i in range(self.ar+self.ma+self.omega+self.beta,
                           self.ar+self.ma+self.beta+self.omega*2):
                bnds.append([-np.Inf,np.Inf])
            bnds.append([-np.Inf,np.Inf])
            bnds.append([-np.Inf,0])
        start = time.clock()
        temp = spo.minimize(self.get_loss,self.pars ,bounds = bnds, method = 'L-BFGS-B')
        end = time.clock()
        print(str(end-start)+' secs used')
        if not temp['success']:
            print('Warn: optimization failed')
            
        self.pars = temp['x']
        self.get_loss(self.pars,True)
        return temp
        
    def get_next(self,ahead = 5):
        x = np.concatenate((self.x_series,np.repeat(0,ahead)))
        ars,mas,betas,omegas,lnys,ur,uo = self.get_pars(self.pars)
        if self.mid == False:
            ur,uo = 0,0
        hlen = self.hlen
        
        #get ep
        ep = np.zeros(hlen + ahead)
        rp = np.zeros(hlen + ahead)
        if self.ar == 0 and self.ma == 0:
            ep = x - ur
            rp = np.repeat(ur,hlen + ahead)
        else:
            for i in range(self.start_loc,hlen + ahead):
                rp[i] = ur + x[(i-self.ar):i].dot(ars) + ep[(i-self.ma):i].dot(mas)
                if i < hlen:
                    ep[i] = x[i] - rp[i]

        #get op
        op = np.ones(hlen + ahead) * self.var0
        if self.beta == 0 and self.omega == 0:
            op = uo
        else:
            for i in range(self.start_loc,hlen + ahead):
                tp = (np.abs(ep[(i-self.omega):i])+lnys*ep[(i-self.omega):i])/\
                     np.sqrt(op[(i-self.beta):i])
                op[i] = np.exp(uo + np.log(op[(i-self.beta):i]).dot(betas) + tp.dot(omegas))
        
        res_rp = rp[hlen:]
        res_op = np.sqrt(op[hlen:])
        
        return res_rp,res_op
        
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
    
    import tushare as ts
    data = ts.get_h_data('000001',index = True,start = '2014-01-01')
    data = data.sort()
    price = data['close'].values*1
    ret = price[1:]/price[:-1]-1
    
    x = ret[-100:]
    egarch = EGarch(x,0,0,1,1,seed = 12221)
    egarch.train()
    plt.plot(garch.sigma)
    print(np.sum(np.abs(x)>garch.sigma*1.96)/len(x))
    rp,op = garch.get_next()
    
    