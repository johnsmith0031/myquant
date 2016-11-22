import numpy as np
import scipy.optimize as spo
import scipy.stats as sps
from myquant.derivative import *
from myquant.fracdiff import *
import time

class fiGarch():
    
    def __init__(self, x, ar = 1, ma = 1, beta = 1, omega = 1, depth = 100, di_fixed = None, mid = True,seed = 1000):
        self.x = x
        self.ar = ar
        self.ma = ma
        self.beta = beta
        self.omega = omega
        self.depth = depth
        self.di_fixed = di_fixed
        rng = np.random.RandomState(seed)
        self.pars = rng.uniform(0,1,ar+ma+beta+omega+3)
        self.pars[ar+ma] = 0.8
        self.pars[ar+ma+beta] = 0.1
        self.pars[-3] = 0
        self.pars[-2] = 0
        self.pars[-1] = 0
        self.var0 = np.std(x)**2
        self.mid = mid
        self.frac = Fracdiff([])
        
    def get_last(self, data, n, v = 0):
        res = []
        for i in data[::-1]:
            if len(res) >= n:
                break
            res.append(i*1)
        while len(res) < n:
            res.append(v)
        return np.array(res)
    
    def get_pars(self,pars):
        loc = 0
        ars = pars[loc:(loc+self.ar)]
        loc += self.ar
        mas = pars[loc:(loc+self.ma)]
        loc += self.ma
        betas = pars[loc:(loc+self.beta)]
        loc += self.beta
        omegas = pars[loc:(loc+self.omega)]
        di = pars[-3]
        ur = pars[-2]
        uo = pars[-1]
        
        return ars,mas,betas,omegas,di,ur,uo
        
    def diff(self,data,di):
        res = self.frac.diff(data,di,self.depth)
        return res
    
    def init_series(self):
        max_loc = np.max([self.ar,self.ma,self.beta,self.omega,self.depth])
        self.start_loc = max_loc
        self.x_series = np.concatenate((np.repeat(0,max_loc),self.x))
        self.hlen = len(self.x_series)
    
    def get_loss(self,pars,renew_params = False):
        
        #load params
        ars,mas,betas,omegas,di,ur,uo = self.get_pars(pars)
        if self.di_fixed is not None:
            di = self.di_fixed
        x = self.diff(self.x_series,di)
        if self.mid == False:
            ur,uo = 0,0
        hlen = self.hlen
        
        #get ep
        ep = np.zeros(hlen)
        rp = np.zeros(hlen)
        if self.ar == 0 and self.ma == 0:
            ep[self.start_loc:] = (x - ur)[self.start_loc:]
            rp[self.start_loc:] = ur
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
                op[i] = uo + op[(i-self.beta):i].dot(betas) + ep2[(i-self.omega):i].dot(omegas)
        
        #likelihood
        L = np.sum(np.log(op[self.start_loc:])/2 + ep2[self.start_loc:]/(2*op[self.start_loc:]))
        
        if renew_params:
            self.sigma = np.sqrt(op[self.start_loc:])
            self.rps = rp[self.start_loc:]
            self.res = ep[self.start_loc:]
            
        return L
        
    def train(self,bounds = True, prints = True):
        self.init_series()
        bnds = None
        if bounds:
            bnds = []
            for i in range(0,self.ar+self.ma):
                bnds.append([-np.Inf,np.Inf])
            for i in range(self.ar+self.ma,len(self.pars)-3):
                bnds.append([0,np.Inf])
            bnds.append([-0.5,0.5])
            bnds.append([-np.Inf,np.Inf])
            bnds.append([0,np.Inf])
        start = time.clock()
        temp = spo.minimize(self.get_loss,self.pars , bounds = bnds, method = 'L-BFGS-B')
        end = time.clock()
        if prints:
            print(str(end-start)+' secs used')
        if not temp['success']:
            print('Warn: optimization failed')
            
        self.pars = temp['x']
        self.get_loss(self.pars,True)
        return temp['x']
    
    def get_next(self,ahead = 5):
        ars,mas,betas,omegas,di,ur,uo = self.get_pars(self.pars)
        if self.di_fixed is not None:
            di = self.di_fixed
        x = self.diff(self.x_series,di)
        x = np.concatenate((x,np.repeat(0,ahead)))
        if self.mid == False:
            ur,uo = 0,0
        hlen = self.hlen
        
        #get ep
        ep = np.zeros(hlen + ahead)
        rp = np.zeros(hlen + ahead)
        if self.ar == 0 and self.ma == 0:
            ep[self.start_loc:] = (x - ur)[self.start_loc:]
            ep[-ahead:] = 0
            rp[self.start_loc:] = ur
        else:
            for i in range(self.start_loc,hlen + ahead):
                rp[i] = ur + x[(i-self.ar):i].dot(ars) + ep[(i-self.ma):i].dot(mas)
                if i < hlen:
                    ep[i] = x[i] - rp[i]

        #get op
        ep2 = ep**2
        op = np.ones(hlen + ahead) * self.var0
        if self.beta == 0 and self.omega == 0:
            op = uo
        else:
            for i in range(self.start_loc,hlen + ahead):
                op[i] = uo + op[(i-self.beta):i].dot(betas) + ep2[(i-self.omega):i].dot(omegas)
        
        res_rp = self.frac.re_diff(rp,di,self.depth)[hlen:]
        res_op = np.sqrt(op[hlen:])
        
        return res_rp,res_op
    
    def get_formers(self):
        
        fdp = self.res/self.sigma
        frp = self.rps
        fop = self.sigma**2
        fss = self.x
        formers = [fop,fdp,frp,fss]
        
        return formers,self.var0
    
    
    def sim(self, rng, sigma0, formers = None): #rng = dp
        ars,mas,betas,omegas,di,ur,uo = self.get_pars(self.pars)
        if self.di_fixed is not None:
            di = self.di_fixed
        if self.mid == False:
            ur,uo = 0,0
        hlen = len(rng)
        
        if formers is not None:
            former_op,former_dp,former_rp,former_series = formers
            
        #get ep
        dp = np.concatenate((np.zeros(self.start_loc),rng))
        op = np.ones(self.start_loc + hlen) * sigma0
        ep = np.zeros(self.start_loc + hlen)
        rp = np.zeros(self.start_loc + hlen)
        if formers is not None:
            rp[:self.start_loc] = former_rp[-self.start_loc:]*1
            dp[:self.start_loc] = former_dp[-self.start_loc:]*1
            op[:self.start_loc] = former_op[-self.start_loc:]*1
            ep[:self.start_loc] = dp[:self.start_loc]*np.sqrt(op[:self.start_loc])
        x = rp+ep
        
        for i in range(self.start_loc,self.start_loc + hlen):
            op[i] = uo + op[(i-self.beta):i].dot(betas) + (ep[(i-self.omega):i]**2).dot(omegas)
            ep[i] = dp[i]*np.sqrt(op[i])
            rp[i] = ur + x[(i-self.ar):i].dot(ars) + ep[(i-self.ma):i].dot(mas)
            x[i] = rp[i] + ep[i]
            
        if formers is not None:
            res = self.frac.re_diff(x[self.start_loc:,],di,self.depth,former_series)
        else:
            res = self.frac.re_diff(x[self.start_loc:,],di,self.depth)
            
        return res
        
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
    
    from myquant.wyshare import *
    data = get_h_data('000905',index = True,start = '20100101')
    close = data['close'].values
    ret = close[1:]/close[:-1]-1
    dates = np.array(data.index,dtype = '<U10')
    
    x = ret[-1001:-1]
    garch = fiGarch(x,0,0,1,1)
    garch.train()
    formers,sigma0 = garch.get_formers()
    
    res = []
    for i in range(1000):
        rng = np.random.randn(240)
        ret2 = garch.sim(rng,sigma0,formers)
        res.append(np.cumprod(1+ret2))
        print(i)
    res = np.array(res).T
    
    plt.plot(res*6341,color = 'grey')
    plt.xlim((0,239))
    
    plt.figure(figsize=(20,8))
    temp = close[-250:]*1
    temp = np.repeat(temp,200).reshape((-1,200))
    temp = np.concatenate((temp,res[:,100:300]*close[-1]),axis=0)
    plt.plot(temp,color = 'red',alpha = 0.1)
    plt.plot(np.repeat(250,100),np.linspace(4000,16000,100),'--',color = 'black',lw = 2,alpha = 0.5)
    for i in range(len(temp)):
        if temp[-1,i] >= 9000 and temp[-1,i] <= 10000:
            plt.plot(temp[:,i],color = 'blue',lw = 2,alpha = 0.5)
            break
    for i in range(len(temp)):
            if temp[-1,i] >= 6500 and temp[-1,i] <= 7500:
                plt.plot(temp[:,i],color = 'blue',lw = 2,alpha = 0.5)
                break
    for i in range(len(temp)):
            if temp[-1,i] >= 4000 and temp[-1,i] <= 5500:
                plt.plot(temp[:,i],color = 'blue',lw = 2,alpha = 0.5)
                break
    plt.plot(np.linspace(0,len(temp)-1,100),np.repeat(close[-1],100),'--',color = 'black', lw = 2,alpha = 0.5)
    plt.xlim((0,len(temp)-1))
    plt.ylim((4000,16000))
    #plt.title('中证500可能走势',fontproperties='SimHei',size = 20)
    #plt.xlabel('日期',fontproperties='SimHei',size = 15)
    
    temp_dates = []
    tps = [31,28,31,30,31,30,31,31,30,31,30,31]
    for year in ['2016','2017']:
        for i in range(12):
            for j in range(tps[i]):
                temp = year + '-'+str(i+1).zfill(2)+'-'+str(j+1).zfill(2)
                if time.strftime('%a',time.strptime(temp,'%Y-%m-%d')) not in ['Sat','Sun']:
                    temp_dates.append(temp)
    temp_dates = np.array(temp_dates)
    dates_final = np.concatenate((dates[-250:],temp_dates[157:(157+240)]))
    
    dts = np.array([i[-8:] for i in dates_final],dtype='<U10')
    nums = 10
    interval = len(dts) // nums
    loc = len(dts) - np.array(range(int(len(dts)/interval)+1))*interval - 1
    if loc[-1] == -1:
        loc[-1] = 0
    plt.xlim((0,len(dts)-1))
    plt.xticks(loc, dts[loc])
    
    def gaussian_kernel(x):
        res = np.exp(-x**2/2)/(np.sqrt(np.pi*2))
        return res
    
    def kernel_density(x,data,window = 1):
        res = []
        for k in x:
            res.append(1/window*gaussian_kernel((k-data)/window).sum()/len(data))
        return np.array(res)
    
    from myquant.quant_ratio import *
    
    mdd = []
    ret_adjust = np.cumprod(np.repeat(1.08**(1/240),240))
    for i in range(1000):
        mdd.append(MDB(res[:,i]*ret_adjust))
    mdd = np.array(mdd)
    plt.figure(figsize = (10,6))
    plt.hist(mdd,bins = 20)
    #plt.title('产品可能最大回撤频数分布直方图',fontproperties='SimHei',size = 15)
        
    density_x = np.linspace(0,1,101)
    density_y = kernel_density(density_x,mdd,0.03)
    cum_density_y = np.cumsum(density_y*0.01)
 
    fig,axes = plt.subplots(1,2,figsize = (20,4))
    axes[0].plot(density_x,density_y)
    axes[1].plot(density_x,cum_density_y)
    axes[1].plot(density_x,np.repeat(0.8,len(density_x)),'g--',lw=2)
    #axes[0].set_title('产品可能最大回撤概率密度函数',fontproperties='SimHei',size = 15)
    #axes[1].set_title('产品可能最大回撤经验分布函数',fontproperties='SimHei',size = 15)
    axes[0].set_ylabel('概率密度',fontproperties='SimHei',size = 15)
    axes[1].set_ylabel('概率',fontproperties='SimHei',size = 15)
    axes[0].grid()
    axes[1].grid()


    plt.figure(figsize = (10,6))
    plt.hist(res[-1,:]*close[-1],bins = 30,color = 'red')
    #plt.title('中证500一年后可能点位频数分布直方图',fontproperties='SimHei',size = 15)
    
    density_x = np.linspace(2000,20000,1001)
    density_y = kernel_density(density_x,res[-1,:]*close[-1],700)
    cum_density_y = np.cumsum(18*density_y)
    
    fig,axes = plt.subplots(1,2,figsize = (20,4))
    axes[0].plot(density_x,density_y)
    axes[1].plot(density_x,cum_density_y)
    #axes[1].plot(density_x,np.repeat(0.2,len(density_x)),'g--',lw=2)
    axes[0].plot(np.repeat(close[-1]*0.92,100),np.linspace(0,0.0002,100),'g--',lw=2)
    axes[1].plot(np.repeat(close[-1]*0.92,100),np.linspace(0,1,100),'g--',lw=2)
    for i in range(1000):
        if density_x[i] >= close[-1]*0.92:
            break
    axes[1].plot(density_x,np.repeat(cum_density_y[i],1001),'g--',lw=2)
    #axes[0].plot(density_x,np.repeat(density_y[i],1001),'g--',lw=2)
    #axes[0].set_title('中证500一年后可能点位概率密度函数',fontproperties='SimHei',size = 15)
    #axes[1].set_title('中证500一年后可能点位经验分布函数',fontproperties='SimHei',size = 15)
    axes[0].set_ylabel('概率密度',fontproperties='SimHei',size = 15)
    axes[1].set_ylabel('概率',fontproperties='SimHei',size = 15)
    axes[0].grid()
    axes[1].grid()
    axes[0].set_ylim((0,0.0002))
    
    temp = res[-1,:]*close[-1]
    temp_y = np.random.rand(len(temp))*0.01
    plt.plot(temp,temp_y,'r.',alpha=0.5)
    
    
    
    
    
    import scipy.stats.kde as kde
    test=kde.gaussian_kde(res[-1,:]*close[-1])
    density_x = np.linspace(2000,16000,1000)
    density_y = test.pdf(density_x)
    
    
#    ress = garch.res
#    frac = Fracdiff(ress)
#    frac._acfs(ress,10)
    
#    sigma1 = garch.sigma
#    plt.plot(sigma1)
#    rng = garch.res/garch.sigma
    
    rng = (garch.res/garch.sigma)[150:]
    fdp = (garch.res/garch.sigma)[:150]
    frp = garch.rps[:150]
    fop = (garch.sigma**2)[:150]
    fss = x[:150]
    formers = [fop,fdp,frp,fss]
    
    ret2 = garch.sim(rng,garch.var0,formers)
    ret2-x[150:]
    
    plt.plot(garch.sim(rng,garch.var0))
    
    
    import scipy.stats as sts
    sts.kstest(rng,'norm')
    
    
    x = ret[-300:]
    garch = fiGarch(x,1,0,1,1)
    garch.train()
    rp,op = garch.get_next()
    di = garch.pars[-3]
    sigma2 = garch.sigma
    plt.plot(sigma2)
    sigma,p_values = garch.get_param_stats()
    
    print(np.sum(np.abs(x-garch.rps)>garch.sigma*2.56)/len(x))
    
    
    
    
    