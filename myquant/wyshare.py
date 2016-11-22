import urllib
import pandas as pd
import numpy as np
import tushare as ts
import time

def get_page(url):  #获取页面数据
    req=urllib.request.Request(url,headers={
        'Connection': 'Keep-Alive',
        'Accept': 'text/html, application/xhtml+xml, */*',
        'Accept-Language':'zh-CN,zh;q=0.8',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko'
    })
    opener=urllib.request.urlopen(req)
    page=opener.read()
    return page

def get_fenhong(code, peigu_flag = False):
    url = 'http://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/'+ code +'.phtml'
    page=get_page(url).decode('gb2312')
    
    temp = page.split('<!--分红 begin-->')[1].split('<!--分红 end-->')[0]
    temp = temp.replace('\t','').replace('\r','').replace(' ','')
    temp = temp.split('\n')
    res = []
    for i in temp:
        if '<td>' in i:
            res.append(i[4:-5])
    hlen = len(res)
    if hlen % 9 != 0:
        raise Exception('数据格式错误')
    res = np.array(res).reshape((hlen//9,9))
    res = res[:,:-1]
    res = pd.DataFrame(res)
    res.columns = ['公告日期','送股','转增','派息','进度','除权除息日','股权登记日','红股上市日']
    for key in ['送股','转增','派息']:
        res[key] = res[key].astype('float')
    res = res.sort('公告日期')
    res.index = np.array(range(len(res)))
    fenhong = res
    
    if peigu_flag:
        temp = page.split('<!--配股 begin-->')[1].split('<!--配股 end-->')[0]
        temp = temp.replace('\t','').replace('\r','').replace(' ','')
        temp = temp.split('\n')
        res = []
        for i in temp:
            if '<td>' in i:
                res.append(i[4:-5])
        hlen = len(res)
        if hlen % 11 != 0:
            raise Exception('数据格式错误')
        res = np.array(res).reshape((hlen//11,11))
        res = res[:,:-2]
        res = pd.DataFrame(res)
        res.columns = ['公告日期','每10股配股股数','配股价格','基准股本','除权日','股权登记日','缴款起始日','缴款终止日','配股上市日']
        for key in ['每10股配股股数','配股价格','基准股本']:
            res[key] = res[key].astype('float')
        res = res.sort('公告日期')
        res.index = np.array(range(len(res)))
        peigu = res
    
        return fenhong,peigu
    else:
        return fenhong
    
def get_h_data(code = '', index = False, start = '', end = '', fuquan = True, pause = 0.01):
    """
    :param index_temp: for example, 'sh000001' 上证指数
    :return:
    """    
    time.sleep(pause)
    if code == '':
        raise Exception('Please Input Code')
    if start == '':
        start = '20100101'
    if end == '':
        end = time.strftime('%Y%m%d')
    
    if index:
        index_id = '0' + code
        fuquan = False
    else:
        if code < '600000':
            index_id = '1' + code
        else:
            index_id = '0' + code
    if not index:
        url='http://quotes.money.163.com/service/chddata.html?code=%s&start=%s&end=%s&fields=TOPEN;HIGH;LOW;TCLOSE;LCLOSE;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP'%(index_id,start,end)
    else:
        url='http://quotes.money.163.com/service/chddata.html?code=%s&start=%s&end=%s&fields=TOPEN;HIGH;LOW;TCLOSE;LCLOSE;VOTURNOVER;VATURNOVER'%(index_id,start,end)

    page=get_page(url).decode('gb2312') #该段获取原始数据
    page=page.split('\r\n')
    
    if not index:
        col_info = ['date','code','name','open','high','low','close','lastclose','turnoverRate','turnoverVol','turnoverValue','TCAP','MCAP']
    else:
        col_info = ['date','code','name','open','high','low','close','lastclose','volume','amount']
        
    index_data=page[1:-1]
    
    for i in range(len(index_data)):
        index_data[i] = index_data[i].replace('\'','')
        index_data[i] = index_data[i].split(',')
    
    df = pd.DataFrame(index_data)
    df.columns = col_info
    df = df.set_index(col_info[0])
    df = df.sort()
    loc = df.values[:,2:] == 'None'
    df.values[:,2:][loc] = '0.0'
    df.values[:,2:] = np.array(df.values[:,2:],dtype = 'float')
    if np.isnan(np.array(df['close'].values,dtype = float)).any():
        raise Exception('data error')
    loc = (df['close'].values == 0)
    for i in range(2,6):
        df.values[loc,i] = df.values[loc,6]
    
    if fuquan:
        fac = ts.stock.trading._parase_fq_factor(code,'','')
        fac = fac.sort(['date'])
        fac['date'] = fac['date'].astype('<U10')
        fac = fac.set_index('date')
        fac = fac.reindex(df.index)
        fac = fac.fillna(method = 'ffill').fillna(method = 'bfill')
        factors = (fac['factor']/df['close']).values*1
        df['factors'] = factors
        
    for col in df.columns[2:]:
        df[col] = df[col].astype('float')
    
    return df

if __name__ == '__main__':
    
    
    codes = ['000001','600415','601818','300024']
    

    data = get_h_data(code = '511990', start = '20140101')
    data['date'] = data.index
    
    import os
    import traceback as tb
    #path = 'T:\\Python Work\\wyshare\\data'
    path = 'C:\\Users\\oqys\\Desktop\\T+1交割单\\data'
#    eq = ts.Equity()
#    temp = eq.Equ('A',field='ticker').values.flatten()
#    codes = [str(i).zfill(6) for i in temp]
    data = get_h_data('601818', start = '20140101', index = False)
    data.to_csv('C:\\Users\\oqys\\Desktop\\T+1交割单\\index.csv')
    def download(code):
        if os.path.exists(path+'\\'+code+'.csv'):
            print(code+' ok')
            return
        try:
            data = get_h_data(code = code, start = '20140101',fuquan = False)
            data.to_csv(path+'\\'+code+'.csv')
            print(code+' saved')
        except:
            print(code + ' failed')
                
                
    from myquant.multithread import *
    
    multi_thread_gevent(download,codes,10)
    