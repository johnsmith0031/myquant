import numpy as np
import scipy.optimize as spo   

class NN():
    
    def __init__(self,n,p,hidden,ep):
        self.ep = ep
        self.n, self.p = n, p
        self.hidden = hidden
        W = self.initHW()
        self.par = W
                        
    def initHW(self):
        p, hidden = self.p, self.hidden
        wlen = 0
        for i in range(len(hidden)):
            if i == 0:
                wlen = wlen + p*hidden[0]
            else: 
                wlen = wlen + hidden[i-1]*hidden[i]
        wlen = wlen + hidden[-1]
        wlen = wlen + np.sum(hidden) + 1
        W = np.random.rand(wlen,1)
        return W
       
    def core(self,x):
        res = np.exp(-x**2/2)
        return res
        
    def dcore(self,x):
        res = np.exp(-x**2/2)*(-x)
        return res
        
    def get_outcome(self,x):
        hidden = self.hidden
        n = x.shape[0]
        hlen = len(hidden)
        temp_W = self.get_W(self.par)
        w = temp_W[:(hlen+1)]
        b = temp_W[(hlen+1):]
        a = []
        z = []
        a.append(x)
        z.append(a[0])
        for i in range(hlen):
            a.append(z[i].dot(w[i]) + np.repeat(b[i],n).reshape((hidden[i],n)).T)
            z.append(self.core(a[i+1]))
        a.append(z[hlen].dot(w[hlen]) + b[hlen])
        z.append(a[hlen+1])
        return z[hlen+1]
        
    def get_loss(self,x,y,W):
        hidden = self.hidden
        n = x.shape[0]
        hlen = len(hidden)
        temp_W = self.get_W(W)
        w = temp_W[:(hlen+1)]
        b = temp_W[(hlen+1):]
        a = []
        z = []
        a.append(x)
        z.append(a[0])
        for i in range(hlen):
            a.append(z[i].dot(w[i]) + np.repeat(b[i],n).reshape((hidden[i],n)).T)
            z.append(self.core(a[i+1]))
        a.append(z[hlen].dot(w[hlen]) + b[hlen])
        z.append(a[hlen+1])
        loss = 0.5*((z[hlen+1] - y)**2).sum() + 0.5*self.ep*W.T.dot(W)
        return loss
                    
    def get_W(self,W):
        p, hidden = self.p, self.hidden
        res = []
        hlen = len(hidden)
        for i in range(hlen):
            if i == 0:
                res.append(W[0:(p*hidden[0]),0].reshape((p,hidden[0])))
                loc = (p*hidden[0]-1)
            else:
                res.append(W[(loc+1):(loc+hidden[i-1]*hidden[i]+1),0].reshape((hidden[i-1],hidden[i])))
                loc = loc + hidden[i-1]*hidden[i]
        res.append(W[(loc+1):(loc+hidden[hlen-1]+1),0].reshape((hidden[hlen-1],1)))
        loc = loc + hidden[hlen-1]
        for i in range(hlen):
            res.append(W[(loc+1):(loc+hidden[i]+1)].reshape((hidden[i],1)))
            loc = loc + hidden[i]
        res.append(W[loc+1].reshape((1,1)))
        return res
        
    def train(self,x,y):
        W = self.initHW()
        par = W.reshape(len(W))
        def func(par):
            res = self.get_loss(x,y,par.reshape((len(par),1)))[0,0]
            return res
        res = spo.minimize(func,par,method="BFGS")
        self.par = res['x']
        self.par = self.par.reshape((len(self.par),1))
        print('complete')
        return res