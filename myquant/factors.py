from myquant.finance import *
from myquant.Hurst import *
import time

def get_new_factors(high,low,close,index_close,amount,tr):
    hlen = len(amount)
    acc_pos = 0
    pos = np.array(np.zeros(600)-1,dtype = int)
    for i in range(len(amount)):
        if amount[hlen-1-i] != 0:
            acc_pos += 1
        if acc_pos != 0:
            pos[acc_pos-1] = hlen-1-i
        if acc_pos >= 600:
            break
    start = time.clock()
    if pos[14] != -1:
        MFIexp = get_MFI(high[pos[14]:],low[pos[14]:],close[pos[14]:],amount[pos[14]:])[-1]
        ATR14 = get_ATR14(high[pos[14]:],low[pos[14]:],close[pos[14]:],amount[pos[14]:])[-1]
    else:
        MFIexp,ATR14 = np.nan,np.nan
    if pos[20] != -1:
        ILLIQUIDITY = get_ILLIQUIDITY(close[pos[20]:],amount[pos[20]:])[-1]
        Skewness = get_Skewness(close[pos[20]:],amount[pos[20]:])[-1]
        CCI20 = get_CCI20(high[pos[20]:],low[pos[20]:],close[pos[20]:],amount[pos[20]:])[-1]
        MTMMA = get_MTMMA(close[pos[20]:],amount[pos[20]:])[-1]
        BIAS20 = get_BIAS20(close[pos[20]:],amount[pos[20]:])[-1]
        VOL20 = get_VOL20(tr[pos[20]:],amount[pos[20]:])[-1]
        REVS20 = get_REVS20(close[pos[20]:],amount[pos[20]:])[-1]
    else:
        ILLIQUIDITY,Skewness,CCI20,MTMMA,BIAS20,VOL20,REVS20 = np.repeat(np.nan,7)
    if pos[60] != -1:
        BIAS60 = get_BIAS60(close[pos[60]:],amount[pos[60]:])[-1]
        VOL60 = get_VOL60(tr[pos[60]:],amount[pos[60]:])[-1]
    else:
        BIAS60,VOL60 = np.nan,np.nan
    if pos[65] != -1:
        DHILO = get_DHILO(high[pos[65]:],low[pos[65]:],amount[pos[65]:])[-1]
    else:
        DHILO = np.nan
    if pos[80] != -1:
        Hurstexp = get_Hurst(close[pos[80]:],amount[pos[80]:])[-1]
    else:
        Hurstexp = np.nan
    if pos[88] != -1:
        CCI88 = get_CCI88(high[pos[88]:],low[pos[88]:],close[pos[88]:],amount[pos[88]:])[-1]
    else:
        CCI88 = np.nan
    if pos[120] != -1:
        DAVOL20 = get_DAVOL20(tr[pos[120]:],amount[pos[120]:])[-1]
    else:
        DAVOL20 = np.nan
    if pos[240] != -1:
        RSTR12 = get_RSTR12(close[pos[240]:],amount[pos[240]:])[-1]
    else:
        RSTR12 = np.nan
    if pos[247] != -1:
        HBETA,HSIGMA = get_HBETA_HSIGMA(close[pos[247]:],index_close[pos[247]:],amount[pos[247]:])
        HBETA = HBETA[-1]
        HSIGMA = HSIGMA[-1]
    else:
        HBETA,HSIGMA = np.nan,np.nan
    if pos[480] != -1:
        RSTR24 = get_RSTR12(close[pos[480]:],amount[pos[480]:])[-1]
    else:
        RSTR24 = np.nan
    end = time.clock()
    print(end-start,'sec used')
    
    res = [MFIexp,ILLIQUIDITY,Skewness,CCI20,MTMMA,BIAS20,VOL20,BIAS60,VOL60,\
            Hurstexp,CCI88,RSTR12,HBETA,HSIGMA,RSTR24,DHILO,DAVOL20,ATR14,REVS20]
    return res


#完全挖掘
def get_ILLIQUIDITY(close,amount):
    ret = np.concatenate((np.zeros(1),close[1:]/close[:-1]-1))
    res = [amount[0]*1]
    old_amount = amount[0]*1
    if old_amount == 0:
        loc = [0]
    else:
        loc = []
    for i in range(1,len(ret)):
        if amount[i] == 0:
            if old_amount == 0:
                res.append(np.nan)
            else:
                res.append(old_amount*1)
            loc.append(i)
            continue
        if ret[i] >= 0.095 or ret[i] <= -0.095:
            if amount[i] >= old_amount:
                res.append(amount[i]*1)
            else:
                res.append(old_amount*1)
        else:
            res.append(amount[i]*1)
            old_amount = amount[i]*1
    ret = ret[amount!=0]
    amount = np.array(res)[amount!=0]
    rate = np.nan_to_num(np.abs(ret)/amount)*1e9
    rate = MA(rate).ma(20)
    rate[:20] = np.nan
    rate = rate.tolist()
    for i in loc:
        if i == 0:
            rate.insert(i,np.nan)
        else:
            rate.insert(i,rate[i-1]*1)
    rate = np.array(rate)
    return rate

#不完全挖掘
def get_Skewness(close,amount):
    loc = amount != 0
    close = close[loc]
    res = np.repeat(np.nan,20).tolist()
    ex = MA(close).ma(20)
    ex2 = MA(close**2).ma(20)
    ex3 = MA(close**3).ma(20)
    for i in range(20,len(close)):
        skew = (ex3[i]-3*ex[i]*ex2[i] + 2*ex[i]**3)/(ex2[i] - ex[i]**2)**(3/2)
        res.append(skew)
    for i in range(len(amount)):
        if amount[i] == 0:
            if i == 0:
                res.insert(i,np.nan)
            else:
                res.insert(i,res[i-1]*1)
    return np.array(res)

#自我流因子
def get_MTMMA(close,amount):
    loc = amount != 0
    close = close[loc]
    mtmma = MTM(close).mtm(8,11)
    mtmma[:20] = np.nan
    mtmma /= close
    mtmma = mtmma.tolist()
    for i in range(len(amount)):
        if amount[i] == 0:
            if i == 0:
                mtmma.insert(i,np.nan)
            else:
                mtmma.insert(i,mtmma[i-1]*1)
    return np.array(mtmma)

#待定待优化相关系数0.9
def get_Hurst(close,amount):
    loc = amount != 0
    close = close[loc]
    ret = np.concatenate((np.zeros(1),np.log(close[1:]/close[:-1])))
    hurst = Hurst(ret)
    res = np.repeat(np.nan,80).tolist()
    for i in range(80,len(ret)):
        res.append(hurst.hurst(i-80,i))
    for i in range(len(amount)):
        if amount[i] == 0:
            if i == 0:
                res.insert(i,np.nan)
            else:
                res.insert(i,res[i-1]*1)
    return np.array(res)

#待定待优化相关系数0.9
def get_HBETA_HSIGMA(close,index_close,amount):    
    loc = amount != 0
    close = close[loc]
    index_close = index_close[loc]
    ret = np.concatenate((np.zeros(1),close[1:]/close[:-1]-1))
    index_ret = np.concatenate((np.zeros(1),index_close[1:]/index_close[:-1]-1))
    hlen = len(ret)
    res = np.repeat(np.nan,247).tolist()
    betas = np.repeat(np.nan,247).tolist()
    if hlen <= 247:
        raise Exception('all value is nan')
    xt = np.column_stack((np.ones(len(ret)-3),ret[:-3],ret[1:-2],ret[2:-1],index_ret[1:-2],index_ret[2:-1],index_ret[3:]))
    yt = ret[3:].reshape((-1,1))
    for i in range(hlen-247):
        if i == 0:
            X = xt[(-244-i):]
            Y = yt[(-244-i):]
        else:
            X = xt[(-244-i):-i]
            Y = yt[(-244-i):-i]
        try:
            beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        except:
            beta = np.linalg.inv(X.T.dot(X) + np.eye(7)*0.0001).dot(X.T).dot(Y)
        ress = Y - X.dot(beta)
        res.insert(247,np.std(ress))
        betas.insert(247,beta[-1,0])
    for i in range(len(amount)):
        if amount[i] == 0:
            if i == 0:
                res.insert(i,np.nan)
                betas.insert(i,np.nan)
            else:
                res.insert(i,res[i-1]*1)
                betas.insert(i,betas[i-1]*1)
    return np.array(betas),np.array(res)

#完全挖掘
def get_RSTR12(close,amount):
    loc = amount != 0
    close = close[loc]
    rs = RSI(close).rs(240)
    rs = 2*(rs-1)
    rs[:240] = np.nan
    rs = rs.tolist()
    for i in range(len(amount)):
        if amount[i] == 0:
            if i == 0:
                rs.insert(i,np.nan)
            else:
                rs.insert(i,rs[i-1]*1)
    return np.array(rs)

#完全挖掘
def get_RSTR24(close):
    loc = amount != 0
    close = close[loc]
    rs = RSI(close).rs(480)
    rs = 2*(rs-1)
    rs[:480] = np.nan
    rs = rs.tolist()
    for i in range(len(amount)):
        if amount[i] == 0:
            if i == 0:
                rs.insert(i,np.nan)
            else:
                rs.insert(i,rs[i-1]*1)
    return rs

#完全挖掘
def get_MFI(high,low,close,amount):
    loc = amount != 0
    h1,l1,c1,m1 = high[loc],low[loc],close[loc],amount[loc]
    mfi = MFI(h1,l1,c1,m1).mfi(14)
    mfi[:14] = np.nan
    mfi = mfi.tolist()
    for i in range(len(amount)):
        if amount[i] == 0:
            if i == 0:
                mfi.insert(i,np.nan)
            else:
                mfi.insert(i,mfi[i-1]*1)
    return np.array(mfi)

#完全挖掘
def get_BIAS20(close,amount):
    loc = amount != 0
    close = close[loc]
    ma20 = MA(close).ma(20)
    bias20 = (close - ma20)/ma20*100
    bias20[:20] = np.nan
    bias20 = bias20.tolist()
    for i in range(len(amount)):
        if amount[i] == 0:
            if i == 0:
                bias20.insert(i,np.nan)
            else:
                bias20.insert(i,bias20[i-1]*1)
    return np.array(bias20)

#完全挖掘
def get_BIAS60(close,amount):
    loc = amount != 0
    close = close[loc]
    ma60 = MA(close).ma(60)
    bias60 = (close - ma60)/ma60*100
    bias60[:60] = np.nan
    bias60 = bias60.tolist()
    for i in range(len(amount)):
        if amount[i] == 0:
            if i == 0:
                bias60.insert(i,np.nan)
            else:
                bias60.insert(i,bias60[i-1]*1)
    return np.array(bias60)

#完全挖掘
def get_CCI20(high,low,close,amount):
    loc = amount != 0
    h1,l1,c1 = high[loc],low[loc],close[loc]
    cci20 = CCI(h1,l1,c1).cci(20)
    cci20[:20] = np.nan
    cci20 = cci20.tolist()
    for i in range(len(amount)):
        if amount[i] == 0:
            if i == 0:
                cci20.insert(i,np.nan)
            else:
                cci20.insert(i,cci20[i-1]*1)
    return np.array(cci20)

#完全挖掘
def get_CCI88(high,low,close,amount):
    loc = amount != 0
    h1,l1,c1 = high[loc],low[loc],close[loc]
    cci88 = CCI(h1,l1,c1).cci(88)
    cci88[:88] = np.nan
    cci88 = cci88.tolist()
    for i in range(len(amount)):
        if amount[i] == 0:
            if i == 0:
                cci88.insert(i,np.nan)
            else:
                cci88.insert(i,cci88[i-1]*1)
    return np.array(cci88)

#完全挖掘
def get_VOL20(tr,amount):
    loc = amount != 0
    tr = tr[loc]
    res = MA(tr).ma(20)
    res[:20] = np.nan
    res = res.tolist()
    for i in range(len(amount)):
        if amount[i] == 0:
            if i == 0:
                res.insert(i,np.nan)
            else:
                res.insert(i,res[i-1]*1)
    return np.array(res)
    
#完全挖掘
def get_VOL60(tr,amount):
    loc = amount != 0
    tr = tr[loc]
    res = MA(tr).ma(60)
    res[:60] = np.nan
    res = res.tolist()
    for i in range(len(amount)):
        if amount[i] == 0:
            if i == 0:
                res.insert(i,np.nan)
            else:
                res.insert(i,res[i-1]*1)
    return np.array(res)

#完全挖掘
def get_DHILO(high,low,amount):
    loc = amount != 0
    high,low = high[loc],low[loc]
    res = np.repeat(np.nan,65).tolist()
    bv = np.log(high) - np.log(low)
    for i in range((len(bv)-65)):
        if i == 0:
            res.insert(65,np.percentile(bv[-65:],50))
        else:
            res.insert(65,np.percentile(bv[(-65-i):(-i)],50))
    for i in range(len(amount)):
        if amount[i] == 0:
            if i == 0:
                res.insert(i,np.nan)
            else:
                res.insert(i,res[i-1]*1)
    return np.array(res)

#不完全挖掘
def get_DAVOL20(tr,amount):
    loc = amount != 0
    tr = tr[loc]
    res = (MA(tr).ma(20)/MA(tr).ma(120)-1)/100
    res[:120] = np.nan
    res = res.tolist()
    for i in range(len(amount)):
        if amount[i] == 0:
            if i == 0:
                res.insert(i,np.nan)
            else:
                res.insert(i,res[i-1]*1)
    return np.array(res)

#自我流因子
def get_ATR14(high,low,close,amount):
    loc = amount != 0
    high,low,close = high[loc],low[loc],close[loc]
    res = ATR(high,low,close).atr(14)
    res[:14] = np.nan
    res = res/close
    res = res.tolist()
    for i in range(len(amount)):
        if amount[i] == 0:
            if i == 0:
                res.insert(i,np.nan)
            else:
                res.insert(i,res[i-1]*1)
    return np.array(res)
    
#自我流因子
def get_REVS20(close,amount):
    loc = amount != 0
    close = close[loc]
    res = np.zeros(len(close))
    res[20:] = close[20:]/close[:-20] - 1
    res[:20] = np.nan
    res = res.tolist()
    for i in range(len(amount)):
        if amount[i] == 0:
            if i == 0:
                res.insert(i,np.nan)
            else:
                res.insert(i,res[i-1]*1)
    return np.array(res)
    
    
    
    
    
    
    