import numpy as np
import scipy.optimize as spo

class Fracdiff():

    def __init__(self, series):
        self.series = series
    
    def diff(self, series, d = 0.5, depth = 10):
        a = self._diff_params(d,depth)
        series_all = np.concatenate((np.zeros(depth),series))
        res = np.zeros(len(series))
        for i in range(depth+1,len(series_all)+1):
            res[i-depth-1] = np.sum(series_all[(i-depth-1):i]*a)
        return res
        
    def re_diff(self, series, d = 0.5, depth = 10, former_series = None):
        a = self._diff_params(d,depth)
        base = series
        length = len(series)
        if former_series is not None:
            target = np.concatenate((former_series[-(depth+1):],np.zeros(length)))
        else:
            target = np.zeros(length + depth + 1)
        for i in range(depth+1,len(target)):
            target[i] = base[i-depth-1] - target[(i-depth):i].dot(a[:-1])
        res = target[(depth+1):]
        return res
        
    def train(self, depth = 100):
        self._init_loss(depth)
        res = spo.minimize(self._get_loss,0,method = 'L-BFGS-B')
        self.d = res['x']
        return res
    
    def sim(self, d = 0.5, depth = 100, length = 100):
        a = self._diff_params(d,depth)
        target = np.zeros(length + depth + 1)
        base = np.random.randn(length)
        for i in range(depth+1,len(target)):
            target[i] = base[i-depth-1] - target[(i-depth):i].dot(a[:-1])
        res = target[(depth+1):]
        return res
        
    def _acf(self, series, lag):
        sigma2 = np.var(series)
        mu = np.mean(series)
        n = len(series)
        Ecov = np.sum((series[lag:] - mu)*(series[:(n-lag)] - mu))/n
        Ecor = Ecov / sigma2
        return Ecor
    
    def _acfs(self, series, lag_num):
        res = np.zeros(lag_num)
        for i in range(1,lag_num + 1):
            res[i-1] = self._acf(series,i)
        return res
            
    def _init_loss(self, depth):
        self._get_series_all(depth)
        self.depth = depth
    
    def _get_series_all(self,depth):
        self.series_all = np.concatenate((np.zeros(depth),self.series))
    
    def _diff(self, d, depth):
        series_all = self.series_all
        a = self._diff_params(d,depth)
        res = np.zeros(len(self.series))
        for i in range(depth+1,len(series_all)+1):
            res[i-depth-1] = np.sum(series_all[(i-depth-1):i]*a)
        return res
    
    def _diff_params(self, d = 0.5, depth = 10):
        a = np.ones(depth+1)
        a[1] = -d
        for i in range(2,depth+1):
            a[i] = a[i-1] * (d-i+1)/i * (-1)
        return a[::-1]
        
    def _get_loss(self, d):
        res = self._diff(d,self.depth)
        res = res - np.mean(res)
#        acf = self._acfs(res,10)
#        n = len(res)
#        k = np.array(range(len(acf)))+1
#        Q = n*(n+2)*np.sum(acf**2/(n-k))
        ep = res
        Ed = np.sum(ep**2)
        N = len(ep)
        sigma2 = Ed/N
        #likelihood
        L = - N/2*np.log(2*np.pi) - N/2*np.log(sigma2) - 1/sigma2*Ed/2
        return -L

if __name__ == '__main__':
    
    series = np.random.randn(100)
    frac = Fracdiff(series)
    res = frac.diff(series,0.2,100)
    back = frac.re_diff(res,0.2,100)
    
    
    import matplotlib as plt
    
    frac = Fracdiff([])
    temp = frac.sim(0.05,5000,5000)
    temp2 = np.cumsum(temp)
    plt.plot(temp2)
    frac._acf(temp,1)
    
    a = frac._diff_params(0.5,5000)
    
    res = np.ones(1000)
    for i in range(1000):
        series = np.random.randn(1000)
        temp = Fracdiff(series)
        temp.train(20)
        res[i] = temp.d
        print(temp.d)
   
    from myquant.wyshare import *
    data = get_h_data('000001',index = True,start = '20100101')
    close = data['close'].values
    ret = close[1:]/close[:-1]-1
    
    length = 250
    res = np.ones(len(ret)-length)
    for i in range(length,len(ret)):
        temp = Fracdiff(ret[(i-length):i])
        temp.train(300)
        res[i-length] = temp.d
        print(temp.d)
    
    frac = Fracdiff(ret)
    frac.train(5000)
    temp = frac.diff(0.1,1000)
    
    
    dates = data.index
    dts = np.array([i[-5:] for i in dates[(length+1):]],dtype='<U10')
    nums = 20
    interval = len(dts) // nums
    loc = len(dts) - np.array(range(int(len(dts)/interval)+1))*interval - 1
    if loc[-1] == -1:
        loc[-1] = 0
    plt.xlim((0,len(dts)-1))
    plt.xticks(loc, dts[loc])
    plt.plot(res,'r-',lw=2,alpha = 0.5)
    plt.plot(np.ones(len(res))*0,color='black',lw=2,alpha = 0.5)
    plt.twinx().plot(close[(length+1):],'g-',lw=2)
        