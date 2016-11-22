import time
import numpy as np
import pandas as pd
import sqlalchemy
from myquant.functions_z1 import is_in

class StockData():
    
    def __init__(self):
        self.con = sqlalchemy.create_engine('mssql+pyodbc://stock_data:stock_data@127.0.0.1/stock?driver=SQL+Server')
        self.get_all_dates()
        
    def get_all_dates(self):
        sql = 'select distinct tradeDate from HQData where ticker = \'000001\''
        df = pd.io.sql.read_sql_query(sql,self.con)
        self.dates = np.array(df.values.flatten(),dtype = '<U10')
    
    def get_all_codes(self,date,lag = 20):
        s_loc = np.sum(self.dates < date)
        if self.dates[s_loc] > date:
            s_loc -= 1
        sql = 'select distinct ticker from HQData where tradeDate = \''+self.dates[s_loc-lag]+'\''
        df = pd.io.sql.read_sql_query(sql,self.con)
        return df.values.flatten().tolist()
    
    def get_one_stock(self,code):
        df = pd.io.sql.read_sql_query('select tradeDate,\
                               openPrice*accumAdjFactor as openadj,\
                               highestPrice*accumAdjFactor as highadj,\
                               lowestPrice*accumAdjFactor as lowadj,\
                               lowestPrice*accumAdjFactor as closeadj,\
                               turnoverVol as volume\
                               from HQData\
                               where ticker = \''+code+'\' and isOpen = 1\
                               order by tradeDate asc',self.con)
        df.columns = ['date','open','high','low','close','volume']
        data = df
        for i in range(1,6):
            data.loc[data.values[:,i] == 0,data.columns[i]] = np.nan
        data = data.fillna(method = 'ffill')
        return data
    
    def get_close_price(self,codes,date):
        s_loc = np.sum(self.dates < date)
        if self.dates[s_loc] > date:
            s_loc -= 1
        strs = '('
        for key in codes:
            strs += '\''+key+'\','
        strs = strs[:-1]+')'
        sql = 'select ticker,tradeDate,(closePrice*accumAdjFactor) as closeadj from HQData\
                where ticker in '+strs+' and tradeDate = \''+self.dates[s_loc]+'\''
        df = pd.io.sql.read_sql_query(sql,self.con)
        return df

    def get_history(self,codes,date,factor = 'close',num=5):
        s_loc = np.sum(self.dates < date)
        e_loc = s_loc - num + 1
        if self.dates[s_loc] > date:
            s_loc -= 1
            e_loc -= 1
        date_end = self.dates[s_loc]
        date_start = self.dates[e_loc]
        strs = '('
        for key in codes:
            strs += '\''+key+'\','
        strs = strs[:-1]+')'
        if is_in(factor,['close','highest','lowest','open']):
            sql = 'select ticker,tradeDate,('+factor+'Price*accumAdjFactor) as adj from HQData\
                    where ticker in '+strs+' and tradeDate >= \''+date_start+'\' and tradeDate <= \''+date_end+'\'\
                    order by ticker asc,tradeDate asc'
        else:
            sql = 'select ticker,tradeDate,'+factor+' as adj from HQData\
                    where ticker in '+strs+' and tradeDate >= \''+date_start+'\' and tradeDate <= \''+date_end+'\'\
                    order by ticker asc,tradeDate asc'
        df = pd.io.sql.read_sql_query(sql,self.con)
        res = pd.DataFrame()
        for i in range(len(codes)):
            s_loc = i*num
            e_loc = (i+1)*num
            res[df.loc[s_loc,'ticker']] = df.loc[np.array(range(s_loc,e_loc)),'adj'].values
        res.index = df.loc[np.array(range(s_loc,e_loc)),'tradeDate'].values
        return res
    
if __name__ == '__main__':
    data = StockData()
    codes = data.get_all_codes('2014-01-05')[1000:1200]
    start = time.clock()
    data.get_close_price(codes,'2014-01-05')
    end = time.clock()
    print(end-start)
    
    temp = data.get_history(codes,'2015-01-05',10)
    price = temp.values
    
    
    
    
    
    
    
    
    
    
    