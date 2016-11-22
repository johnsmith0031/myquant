import numpy as np
import scipy.optimize as spo
import scipy.stats as sps
import matplotlib.pyplot as plt
from myquant.garch import *

class KernelDensity():
    
    def kernel(self,x):
        res = 1 - np.abs(x)
        res[res<0] = 0
        #res = np.exp(-x**2/2)/np.sqrt(2*np.pi)
        return res
        
    def density(self,x,data,h = 0.1):
        d = x - data
        res = np.mean(self.kernel(d/h)/h)
        return x,res
        
    def plot_density(self,data,color = 'b'):
        lower,upper = np.percentile(data,5),np.percentile(data,95)
        mid = np.percentile(data,50)
        h_range = (upper - lower)*2
        window = h_range/5
        res = np.array([self.density(x,data,window) for x in np.linspace(mid - h_range,mid + h_range,200)])
        plt.plot(res[:,0],res[:,1],color)
    
def lowess(X,Y,x,span = 1/3):
    dis = np.argsort(np.abs(X - x))
    m = int(len(dis)*span)
    X,Y = X[dis[:m]],Y[dis[:m]]
    hi = np.abs(X[-1] - x)
    xt = np.column_stack((np.repeat(1,len(X)),X,X**2))
    temp = (X-x)/hi
    wt = np.zeros_like(temp)
    loc = np.abs(temp)<1
    wt[loc] = (1-np.abs((temp[loc])**3))**3
    wt = wt.reshape((-1,1))
    A = np.array([]).reshape((0,3))
    for i in range(3):
        A = np.row_stack((A,wt.T.dot(xt*X.reshape((-1,1))**i)))
    B = (xt*wt).T
    beta = np.linalg.inv(A).dot(B).dot(Y)
    res = np.array([1,x,x**2]).dot(beta)
    return res
        
def acf(A,B,ranges = (-10,10) , plot = True):
    hlen = len(A)
    a = 2*1.96/np.sqrt(hlen-3)
    bars = np.abs((1-np.exp(a))/(1+np.exp(a)))
    res = []
    for i in range(ranges[0],ranges[1]):
        if i <= 0:
            temp = np.cov(A[:(hlen+i)],B[(-i):hlen])
        else:
            temp = np.cov(A[i:hlen],B[:(hlen-i)])
        r = temp[1,0] / (np.std(A)*np.std(B))
        res.append([i,r])
    res = np.array(res)
    if plot:
        plt.plot(res[:,0],res[:,1],'r-')
        tempx = np.array(range(ranges[0],ranges[1]))
        tempy = np.repeat(bars,len(tempx))
        plt.plot(tempx,tempy,'b--')
        plt.plot(tempx,-tempy,'b--')
        return
    else:
        return res

def linear(x,y,sigma_flag = False,mean = True,rd = 0):
    
    X = np.column_stack((np.repeat(1,len(x)),x))
    if not mean:
        X = X[:,1:]
    Y = y.reshape((-1,1))
    temp_X = np.linalg.inv(X.T.dot(X) + rd * np.eye(X.shape[1]))
    beta = temp_X.dot(X.T).dot(Y)
    res = Y - X.dot(beta)
    sigma = temp_X*np.std(res)**2
    sigma = np.array([np.sqrt(sigma[i,i]) for i in range(len(sigma))])
    t_values = beta.flatten()/sigma
    p_values = 2*(1-sps.t(len(X)-len(beta)).cdf(np.abs(t_values)))
    p_values = np.round(p_values,4)
    R2 = 1 - np.sum(res**2)/np.sum((Y - np.mean(Y))**2)
    
    if sigma_flag:
        return beta.flatten(),sigma,R2,res
        
    return beta.flatten(),p_values,R2,res

def step_foward(x,y,num,fields = None):
    k = 1
    res = y
    X = []
    locs = []
    while k <= num:
        r = np.abs([np.cov(x[:,i],res.flatten())[0,1]/\
                    (np.std(x[:,i])*np.std(res)) for i in range(x.shape[1])])
        loc = np.argmax(r)
        X.append(x[:,loc])
        x = x[:,np.array(range(x.shape[1]))!=loc]
        xt = np.array(X).T
        try:
            beta, p_values,R2,res = linear(xt,y)
        except:
            beta, p_values,R2,res = linear(xt,y,rd = 0.0001)
        k += 1
        if fields is not None:
            locs.append(fields[loc])
            fields = fields[np.array(range(len(fields)))!=loc]
    if fields is None:
        return np.array(X).T
    else:
        return np.array(X).T,locs
        
def beyasian_linear(x,y, alpha = 1, bootstrap = 100):
    
    X = np.column_stack((np.repeat(1,len(x)),x))
    Y = y.reshape((-1,1))
    N = len(X)
    
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    
    def f(w,X,Y):
        w = w.reshape((-1,1))
        res = Y - X.dot(w)
        Ed = (1/2)*np.sum(res**2)
        sigma2 = (2/N)*Ed
        
        L = -N/2*np.log(2*np.pi) - N/2*np.log(sigma2) - (1/sigma2)*Ed -\
            len(w)*np.log(2*np.pi*alpha) - w.T.dot(w)/(2*alpha)
            
        return -L
        
    temp = spo.minimize(f,beta,method = 'L-BFGS-B',args = (X,Y))
    beta = temp['x']
    res = Y - X.dot(beta).reshape((-1,1))
    Ed = (1/2)*np.sum(res**2)
    sigma = np.sqrt((2/N)*Ed)
    
    record = []
    for i in range(bootstrap):
        res_temp = np.random.randn(N)*sigma
        y_temp = (X.dot(beta) + res_temp).reshape((-1,1))
        beta_temp = spo.minimize(f,beta,method = 'L-BFGS-B',args = (X,y_temp))['x']
        record.append(beta_temp.tolist())
    record = np.array(record)
    beta_sigma = np.std(record,axis=0)
    t_values = beta.flatten()/beta_sigma
    p_values = 2*(1-sps.t(len(X)-len(beta)).cdf(np.abs(t_values)))
    p_values = np.round(p_values,4)
    R2 = 1 - np.sum(res**2)/np.sum((Y - np.mean(Y))**2)

    return beta,p_values,R2,res
    
    
def kalman(x, y, s = None, Q = None, H = None, garch = False , mean = True):
    
    s0,E0,R2,ress = linear(x,y,True,mean = mean)
    if H is None:
        if garch:
            garch = Garch(ress.flatten(),0,0,1,1)
            garch.train(prints = False, warn = False)
            H = garch.sigma**2
        else:
            H = np.ones(len(x))*np.array([np.var(ress)])
    if Q is None:
        Q = np.diag(E0**2)
    if s is None:
        s = s0.reshape((-1,1))
    E = Q * 10
    
    x = np.column_stack((np.repeat(1,len(x)),x))
    if not mean:
        x = x[:,1:]
        
    res = []
    sigma = []
    
    for i in range(len(x)):
        Z = x[i].reshape((1,-1))
        v = y[i] - Z.dot(s)
        C = E.dot(Z.T)
        V = Z.dot(C) + H[i]
        s = s + C.dot(np.linalg.inv(V))*v
        E = E - C.dot(np.linalg.inv(V)).dot(C.T)
        sigma.append(np.diag(E))
        E = E + Q
        res.append(s.flatten()*1)
        
    res = np.array(res)
    sigma = np.sqrt(np.array(sigma))
    
    return res,sigma
    
    
def kalman_old(x,y):
    
    temp = linear(x,y,True)
    x = np.column_stack((np.repeat(1,len(x)),x))
    res = np.ones_like(x)
    res[0] = temp[0]
    sigma2_e = np.var(temp[3])
    sigma = np.diag(temp[1]**2)
    for i in range(1,len(x)):
        K = 2*sigma.dot(x[i])/(x[i].T.dot(sigma).dot(x[i])+sigma2_e)
        mu = res[i-1] + K*(y[i]-x[i].T.dot(res[i-1]))
        temp = 2*sigma.dot(x[i]).reshape((-1,1))
        #sigma = sigma - (temp.dot(temp.T))/(x[i].T.dot(sigma).dot(x[i])+sigma2_e)
        res[i] = mu
        i+=1
        if np.isnan(res[i-1]).any():
            print('failed')
            return
            
    return res