import numpy as np

class SV():
    
    def __init__(self,r):
        self.r = r
        self.Et0s = np.zeros(len(r))
        self.Et1s = np.zeros(len(r))
        self.st0s = np.zeros(len(r))
        self.st1s = np.zeros(len(r))
        self.ct,self.Ht = np.zeros(len(r)),np.ones(len(r))
        self.mu_sigma2 = np.array([[-11.4004,5.796],
                                   [-5.2432,2.6137],
                                   [-9.8373,5.1795],
                                   [1.5075,0.1674],
                                   [-0.651,0.6401],
                                   [0.5248,0.3402],
                                   [-2.3586,1.2626]])
        self.prob = np.array([0.00730,
                              0.10556,
                              0.00002,
                              0.04395,
                              0.34001,
                              0.24566,
                              0.25750])
        self.a = np.array([0,1])
        self.b = 0
        self.s02 = 0.01
        self.p = 0
        self.sn2 = 0.01
        #garch = Garch(r,0,0,1,1)
        #garch.train()
        #self.z = 2*np.log(garch.sigma)
        self.z = np.ones(len(r))*np.log(np.var(r))
        
    def sample_b(self):
        hb = np.sqrt(self.s02) * np.exp(self.z/2)
        xt = 1 / hb
        rt = self.r / hb
        sigma = 1/(np.sum(xt**2) + 1)
        mu = sigma*np.sum(xt*rt)
        res = np.random.normal(mu,np.sqrt(sigma))
        self.b = res
        
    def sample_a(self):
        a0 = np.array([-1,1])
        C0 = np.eye(2)
        zt = np.column_stack((np.repeat(1,len(self.z)-1),self.z[:-1]))
        sigma = np.linalg.inv(np.sum(zt**2)/self.sn2 + np.linalg.inv(C0))
        mu = sigma.dot(np.sum(zt*self.z[:-1].reshape((-1,1)))/self.sn2 + np.linalg.inv(C0).dot(a0))
        res = np.random.multivariate_normal(mu,sigma)
        self.a = res
        
    def sample_sn2(self):
        zt = np.column_stack((np.repeat(1,len(self.z)-1),self.z[:-1]))
        vt = self.z[1:] - zt.dot(self.a)
        n = len(self.r)
        m = 10*len(self.r)
        #m = 1
        res = (1 + np.sum(vt[1:]**2))/np.random.chisquare(m+n-1)
        self.sn2 = res
        
    def sample_s02(self):
        vt = (self.r - self.b)*np.exp(-self.z/2)
        n = len(self.r)
        m = 1
        res = (1 + np.sum(vt**2))/np.random.chisquare(n+m)
        self.s02 = res
        
    def sample_sn2_and_p(self):
        y0 = 1
        y1 = 1
        n = len(self.r)
        zt = np.column_stack((np.repeat(1,len(self.z)-1),self.z[:-1]))
        ln = self.z[1:] - zt.dot(self.a)
        e = (self.r - self.b)*np.exp(-self.z/2)/np.sqrt(self.s02)
        e = e[1:]
        w_a = (n+1+y0)/2
        w_b = (y1 + ln.T.dot(ln) - (e.T.dot(ln))**2/(2+e.T.dot(e)))/2
        w = 1 / np.random.gamma(w_a,w_b)
        u_mu = e.T.dot(ln)/(2+e.T.dot(e))
        u_sigma = w / (2+e.T.dot(e))
        u = np.random.normal(u_mu,np.sqrt(u_sigma))
        self.sn2 = w + u**2
        self.p = u/np.sqrt(self.sn2)
        
    def kalman(self,fixed = False):
        
        y = np.log((self.r - self.b)**2/self.s02)
        s = -9
        E = 1
        for i in range(len(y)):
            v_std = (y[i] - s - self.mu_sigma2[:,0])
            status = np.exp(-v_std**2/(2*self.mu_sigma2[:,1]))/np.sqrt(self.mu_sigma2[:,1])*self.prob
            prob = np.cumsum(status/np.sum(status))
            loc = np.sum(np.random.rand()>prob)
            if fixed:
                loc = np.argmax(status)
            self.ct[i],self.Ht[i] = self.mu_sigma2[loc,0],self.mu_sigma2[loc,1]
            v = y[i] - self.ct[i] - s
            C = E
            V = C + self.Ht[i]
            s = s + C/V*v
            E = E - C**2/V
            self.st0s[i] = s*1
            self.Et0s[i] = E*1
            s = self.a[0] + self.a[1]*s
            E = self.a[1]**2*E + self.sn2
            self.st1s[i] = s*1
            self.Et1s[i] = E*1
            
        
    def sample_z(self):
        z = np.concatenate((self.z,np.array([self.st1s[-1]])))
        mu = self.st0s + self.a[1]*self.Et0s/self.Et1s*(z[1:] - self.st1s)
        sigma = self.Et0s - self.a[1]**2*self.Et0s**2/self.Et1s
        self.z = np.random.randn(len(self.z))*sigma + mu
        
    def train(self,burn_in = 100,sample = 1000):
        
        res = []
        for i in range(burn_in):
            self.sample_b()
            self.sample_a()
            self.sample_s02()
            self.sample_sn2()
            res.append([self.b,self.a[0],self.a[1],self.s02,self.sn2])
        res = np.array(res)
        
        res = []
        zs = []
        for i in range(burn_in+sample):
            self.sample_b()
            self.sample_a()
            self.sample_s02()
            self.sample_sn2()
            self.kalman()
            self.sample_z()
            if i >= burn_in:
                res.append([self.b,self.a[0],self.a[1],self.s02,self.sn2])
                zs.append(self.z)
            print(i)
        res = np.array(res)
        zs = np.array(zs)
        
        self.pars = res.mean(axis=0)
        self.pars_sigma = res.std(axis=0)
        self.b,self.a[0],self.a[1],self.s02,self.sn2 = self.pars
        self.sigma = np.exp((zs.mean(axis=0))/2)*np.sqrt(self.s02)
        self.zs = zs
#        self.b,self.a[0],self.a[1],self.s02,self.sn2 = self.pars
#        self.kalman(fixed = True)
#        self.sample_z()
#        self.sigma = np.exp(self.z/2)*np.sqrt(self.s02)
 
if __name__ == '__main__':
    from myquant.garch import *
    import matplotlib.pyplot as plt
    ret = np.random.randn(100)*0.01

    import tushare as ts
    data = ts.get_h_data('000001',index = True,start = '2014-01-01')
    data = data.sort()
    price = data['close'].values*1
    ret = price[1:]/price[:-1]-1

    sv = SV(ret[-100:])
    self = sv
    sv.train()
    garch = Garch(ret,0,0,1,1)
    garch.train()
    plt.plot(sv.sigma,'b-')
    plt.plot(garch.sigma,'g-')
    np.sum(np.abs(ret[-100:])>sv.sigma*1.96)/len(ret[-100:])

    plt.plot(ret[-50:])
    plt.plot(sv.b+sv.sigma)
    plt.plot(sv.b-sv.sigma)