import numpy as np
from myquant.linear_method import kalman

class myMTM():
    
    def __init__(self,close):
        ret = np.concatenate((np.array([0]),close[1:]/close[:-1] - 1))
        self.ret = ret

    def mtm(self,mk_sigma = 0.04,par_sigma = 0.02):
        H = np.ones(len(self.ret))*mk_sigma**2
        Q = np.array([[par_sigma**2]])
        s = np.array([[0]])
        beta,sigma = kalman(np.repeat(1,len(self.ret)),self.ret,mean=False,H=H,Q=Q,s=s)
        res = beta.flatten()
        return res

class MACD():
        
    def __init__(self,prices):
        self.prices = prices
        
    def ema(self,series,alpha):
        res = [series[0]*1]
        for i in range(1,len(series)):
            res.append(res[-1]*(alpha-1)/(alpha+1) + series[i]*2/(alpha+1))
        return np.array(res)
            
    def dif(self):
        res = self.ema(self.prices,12) - self.ema(self.prices,26)
        return res
        
    def dea(self):
        res = self.ema(self.dif(),9)
        return res
            
    def macd(self):
        res = (self.dif() - self.dea()) * 2
        return res

class OBV():
    
    def __init__(self,close,vol):
        self.close = close
        self.vol = vol
        
    def obv(self,N = 20):
        res = [0]
        for i in range(1,len(self.close)):
            if self.close[i-1] < self.close[i]:
                res.append(res[-1] + self.vol[i])
            elif self.close[i-1] > self.close[i]:
                res.append(res[-1] - self.vol[i])
            else:
                res.append(res[-1]*1)
        ma = MA(res).ma(N)
        return ma
        
class ATR():
    
    def __init__(self,high,low,close):
        self.close = close
        self.high = high
        self.low = low
    
    def ema(self,series,alpha):
        res = [series[0]*1]
        for i in range(1,len(series)):
            res.append(res[-1]*(alpha-1)/(alpha+1) + series[i]*2/(alpha+1))
        return np.array(res)
        
    def atr(self, num = 14):
        hlen = len(self.close)
        res = np.zeros(hlen)
        res[0] = self.high[0] - self.low[0]
        for i in range(1,hlen):
            res[i] = np.max([self.high[i] - self.low[i],self.high[i] - self.close[i-1],self.close[i-1] - self.low[i]])
        res = self.ema(res,num)
        return res
        
    def atr_rate(self, num = 14):
        hlen = len(self.close)
        res = np.zeros(hlen)
        res[0] = self.high[0] - self.low[0]
        for i in range(1,hlen):
            res[i] = np.max([self.high[i] - self.low[i],self.high[i] - self.close[i-1],self.close[i-1] - self.low[i]])
        res = self.ema(res,num)
        return res

class MFI():
    
    def __init__(self,high,low,close,amount):
        self.close = close
        self.high = high
        self.low = low
        self.amount = amount
        
    def mfi_slow(self,num = 14):
        typ = (self.close + self.high + self.low) / 3
        res = [50]
        for i in range(1,len(typ)):
            loc = np.max([i-num,0])
            temp = typ[(loc+1):(i+1)] - typ[loc:i]
            vol_temp = self.amount[(loc+1):(i+1)]
            A = vol_temp[temp>0].sum()
            B = vol_temp[temp<0].sum()
            if A + B != 0:
                res.append(A/(A+B)*100)
            else:
                res.append(50)
        return np.array(res)
    
    def mfi(self,num = 14):
        
        typ = (self.close + self.high + self.low) / 3
        buy_sum = 0
        sell_sum = 0
        res = [50]
        for i in range(1,len(typ)):
            if i >= num + 1:
                if typ[i-num] > typ[i-num-1]:
                    buy_sum -= typ[i-num]*self.amount[i-num]
                elif typ[i-num] < typ[i-num-1]:
                    sell_sum -= typ[i-num]*self.amount[i-num]
                    
            if typ[i] > typ[i-1]:
                buy_sum += typ[i]*self.amount[i]
            elif typ[i] < typ[i-1]:
                sell_sum += typ[i]*self.amount[i]
                
            if (buy_sum+sell_sum) == 0:
                res.append(50)
            else:
                res.append(buy_sum/(buy_sum+sell_sum)*100)
        return np.array(res)

class KDJ():
    
    def __init__(self,high,low,close):
        self.close = close
        self.high = high
        self.low = low
        
    def rsv(self,num = 9):
        res = []
        for i in range(len(self.close)):
            if i == 0:
                max = self.high[i]
                min = self.low[i]
            else:
                loc = np.max([i-num+1,0])
                max = np.max(self.high[loc:(i+1)])
                min = np.min(self.low[loc:(i+1)])
            if max == min:
                res.append(50)
            else:
                res.append((self.close[i] - min)/(max - min)*100)
        res = np.array(res)
        return res
        
    def kdj(self,alpha = 1/3,beta = 1/3):
        rsv = self.rsv()
        K = [50]
        D = [50]
        for i in range(len(self.close)):
            K.append(K[-1]*(1-alpha) + rsv[i]*alpha)
            D.append(D[-1]*(1-beta) + K[-1]*beta)
        K = np.array(K)[1:]/100
        D = np.array(D)[1:]/100
        J = 3*K - 2*D
        return K,D,J
        
        
class RSI():
    
    def __init__(self,close):
        self.close = close
    
    def rs(self,num = 9):
        res = [1]
        buy_sum = 0
        sell_sum = 0
        for i in range(1,len(self.close)):
            if i >= num + 1:
                if self.close[i-num] > self.close[i-num-1]:
                    buy_sum -= self.close[i-num] - self.close[i-num-1]
                elif self.close[i-num] < self.close[i-num-1]:
                    sell_sum -= self.close[i-num-1] - self.close[i-num]
                
            if self.close[i] > self.close[i-1]:
                buy_sum += self.close[i] - self.close[i-1]
            elif self.close[i] < self.close[i-1]:
                sell_sum += self.close[i-1] - self.close[i]
                
            if buy_sum == 0 or sell_sum == 0:
                res.append(1)
            else:
                res.append(buy_sum/sell_sum)
        return np.array(res)
    
    def rs_slow(self,num = 9):
        res = [1]
        for i in range(1,len(self.close)):
            loc = np.max([i-num,0])
            dif = self.close[(loc+1):(i+1)] - self.close[loc:i]
            A = dif[dif>0]
            B = -dif[dif<0]
            A = np.nan_to_num(np.sum(A))
            B = np.nan_to_num(np.sum(B))
            if A == 0 or B == 0:
                res.append(1)
            else:
                res.append(A/B)
        return np.array(res)
    
    def rsi(self,num = 9):
        res = [50]
        for i in range(1,len(self.close)):
            loc = np.max([i-num,0])
            dif = self.close[(loc+1):(i+1)] - self.close[loc:i]
            A = dif[dif>0].sum()
            B = -dif[dif<0].sum()
            if A+B == 0:
                res.append(50)
            else:
                res.append(A/(A+B)*100)
        return np.array(res)/100

class EMA():
    
    def __init__(self,close):
        self.close = close

    
    def ema(self,alpha = 5):
        res = [self.close[0]*1]
        for i in range(1,len(self.close)):
            res.append(res[-1]*(alpha-1)/(alpha+1) + self.close[i]*2/(alpha+1))
        return np.array(res)
    
class MA():
    
    def __init__(self,close):
        self.close = close
        
    def ma(self,num = 5):
        res = [self.close[0]*1]
        for i in range(1,len(self.close)):
            if i >= num:
                res.append((res[-1]*num - self.close[i-num]+self.close[i])/num)
            else:
                res.append((res[-1]*i + self.close[i])/(i+1))
        res = np.array(res)
        return res
        
    def msum(self,num = 5):
        res = [self.close[0]*1]
        for i in range(1,len(self.close)):
            if i >= num:
                res.append((res[-1] - self.close[i-num]+self.close[i]))
            else:
                res.append((res[-1] + self.close[i]))
        res = np.array(res)
        return res

class CCI():
    
    def __init__(self,high,low,close):
        self.close = close
        self.high = high
        self.low = low
        
    def cci(self,num = 14):
        typ = (self.close + self.high + self.low) / 3
        ma = MA(typ).ma(num)
        avedev = AVEDEV(typ).avedev(num)
        res = (typ - ma)/(0.015*avedev)
        return res


class AVEDEV():
    
    def __init__(self,close):
        self.close = close
        
    def avedev(self,num = 20):
        ma = MA(self.close).ma(num)
        res = []
        for i in range(1,len(self.close)):
            res.append(np.abs(self.close[np.max([0,i-num]):i] - ma[i]).mean())
        res.insert(0,res[0])
        return np.array(res)

class MTM():
    
    def __init__(self,close):
        self.close = close
        
    def mtm(self,N = 12,M = 6):
        mtm = [0]
        for i in range(1,len(self.close)):
            mtm.append(self.close[i] - self.close[np.max([0,i-N])])
        mtmma = MA(mtm).ma(M)
        return mtmma

class VP():
    
    def __init__(self,close):
        self.close = close
        
    def ema(self,alpha):
        res = [self.close[0]*1]
        for i in range(1,len(self.close)):
            res.append(res[-1]*(alpha-1)/(alpha+1) + self.close[i]*2/(alpha+1))
        return np.array(res)
        
    def vp(self,alpha = 5):
        res = self.close/self.ema(alpha)
        return res


if __name__ == '__main__':
    
    import tushare as ts
    
    data = ts.get_hist_data('sh').sort()
    close = data['close'].values*1
    high = data['high'].values*1
    low = data['low'].values*1
    
    kdj = KDJ(high,low,close)
    K,D,J = kdj.kdj()
    import matplotlib.pyplot as plt
    plt.plot(K)
    plt.plot(D)
    plt.plot(J)
    
    rsi = RSI(close)
    rsi.rsi()
    
    
    
    
    
    
    
    