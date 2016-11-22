import numpy as np
import scipy.optimize as spo
import scipy.stats as sps
from myquant.derivative import *
from myquant.fracdiff import *
import time

class Arfima():
    
    def __init__(self, x, ar = 1, ma = 1, depth = 100, di_fixed = None, mid = True, seed = 1000):
        self.x = x
        self.ar = ar
        self.depth = depth
        self.ma = ma
        self.di_fixed = di_fixed
        rng = np.random.RandomState(seed)
        self.pars = rng.uniform(0,1,ar+ma+2)
        self.pars[-2] = 0
        self.pars[-1] = 0
        self.mid = mid
        self.frac = Fracdiff([])
    
    def get_pars(self,pars):
        loc = 0
        ars = pars[loc:(loc+self.ar)]
        loc += self.ar
        mas = pars[loc:(loc+self.ma)]
        loc += self.ma
        mu = pars[-2]
        di = pars[-1]
        return ars,mas,mu,di
        
    def diff(self,data,di):
        res = self.frac.diff(data,di,self.depth)
        return res
        
    def init_series(self):
        max_loc = np.max([self.ar,self.ma,self.depth])
        self.start_loc = max_loc
        self.x_series = np.concatenate((np.repeat(0,max_loc),self.x))
        self.hlen = len(self.x_series)
    
    def get_loss(self,pars,renew_params = False):
        
        #load params
        ars,mas,mu,di = self.get_pars(pars)
        if self.di_fixed is not None:
            di = self.di_fixed
        x = self.diff(self.x_series,di)
        if self.mid == False:
            mu = 0
        hlen = self.hlen
        
        #get ep
        ep = np.zeros(hlen)
        rp = np.zeros(hlen)
        if self.ar == 0 and self.ma == 0:
            ep[self.start_loc:] = (x - mu)[self.start_loc:]
            rp[self.start_loc:] = mu
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
            self.res = ep[self.start_loc:]
            
        return -L
        
    def train(self,bound = True):
        self.init_series()
        bnds = []
        for i in range(len(self.pars)-1):
            bnds.append([-np.inf,np.inf])
        bnds.append([-0.5,0.5])
        if not bound:
            bnds = None
        start = time.clock()
        temp = spo.minimize(self.get_loss, self.pars, bounds = bnds, method = 'L-BFGS-B')
        end = time.clock()
        print(str(end-start)+' secs used')
        if not temp['success']:
            print('Warn: optimization failed')
        self.pars = temp['x']
        self.get_loss(self.pars,True)
        return temp
 
    def get_next(self,ahead = 5):
        #load params
        ars,mas,mu,di = self.get_pars(self.pars)
        if self.di_fixed is not None:
            di = self.di_fixed
        x = self.diff(self.x_series,di)
        x = np.concatenate((x,np.repeat(0,ahead)))
        if self.mid == False:
            mu = 0
        hlen = self.hlen
        
        #get ep
        ep = np.zeros(hlen+ahead)
        rp = np.zeros(hlen+ahead)
        if self.ar == 0 and self.ma == 0:
            ep[self.start_loc:] = (x - mu)[self.start_loc:]
        else:
            for i in range(self.start_loc,hlen+ahead):
                rp[i] = mu + x[(i-self.ar):i].dot(ars) + ep[(i-self.ma):i].dot(mas)
                if i < hlen:
                    ep[i] = x[i] - rp[i]
        
        res_rp = self.frac.re_diff(rp, di, self.depth)[hlen:]
                
        return res_rp
    
    def get_formers(self):
        fep = self.res
        frp = self.rps
        fss = self.x
        formers = [fep,frp,fss]
        sigma0 = np.std(self.res)
        return formers,sigma0
    
    def sim(self, rng, formers = None):
        ars,mas,mu,di = self.get_pars(self.pars)
        if self.di_fixed is not None:
            di = self.di_fixed
        if self.mid == False:
            mu = 0
        hlen = len(rng)
        
        if formers is not None:
            former_ep,former_rp,former_series = formers
        
        ep = np.zeros(self.start_loc + hlen)
        ep[self.start_loc:] = rng*1
        rp = np.zeros(self.start_loc + hlen)
        if formers is not None:
            rp[:self.start_loc] = former_rp[-self.start_loc:]*1
            ep[:self.start_loc] = former_ep[-self.start_loc:]*1
        x = rp+ep
        for i in range(self.start_loc,hlen + self.start_loc):
            rp[i] = mu + x[(i-self.ar):i].dot(ars) + ep[(i-self.ma):i].dot(mas)
            x[i] = rp[i] + ep[i]
                    
        if formers is not None:
            res = self.frac.re_diff(x[self.start_loc:,],di,self.depth,former_series)
        else:
            res = self.frac.re_diff(x[self.start_loc:,],di,self.depth)
            
        return res
        
        
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
    
    from myquant.wyshare import *
    data = get_h_data('000001',index = True,start = '20140101')
    close = data['close'].values
    ret = close[1:]/close[:-1]-1
    
    x = ret[-200:]
    arma = Arfima(x,0,0,100)
    arma.train()
    formers,sigma0 = arma.get_formers()

    res = []
    for i in range(1000):
        rng = np.random.randn(100)*sigma0
        ret2 = arma.sim(rng,formers)
        res.append(np.cumprod(1+ret2))
        print(i)
    res = np.array(res).T
    plt.plot(res)

    rng = arma.res[120:]
    fep = arma.res[:120]
    frp = arma.rps[:120]
    fss = x[:120]
    formers = [fep,frp,fss]
    
    arma.sim(rng,formers) - x[120:]
    
    sigma,p_values = temp.get_param_stats()
    
    
    di = temp.pars[-1]
    plt.plot(np.cumprod(1+x))
    
    frac = Fracdiff(x)
    frac.train(100)
    
    
    
    
    