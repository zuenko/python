
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats as sts

import sys
import os
import time
import datetime

import logging
from multiprocessing import Pool
from functools import partial

logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s] %(message)s',
                    )


# Загружаем выборки.

# In[2]:


data1 = pd.read_excel('Data_Extract_From_Gender_Statistics.xlsx', encoding = 'utf8')

data2 = pd.read_excel('Data_Extract_From_Health_Nutrition_and_Population_Statistics.xlsx', encoding='utf8')

data3 = pd.read_excel('Data_Extract_From_Millennium_Development_Goals.xlsx', encoding='utf8')

data_cnt = pd.read_csv('all.csv', encoding='utf8')


# Очищаем данные от больших пропусков, точек и пр.

# In[3]:


class Analyst:
       def __init__(self, data, width=100, height=100):
            self.data = data


# In[4]:


def Fillnan(data, years_c):
    for name in years_c:
        data[name] = data[name].apply(lambda x: np.nan if x==('..') else float(x))
    return data
    
def Dropempt(data, years_c):
    tmp=[]
    for ind, item in enumerate(data[years_c].as_matrix()):
        if np.nansum(item) == 0:
            tmp.append(ind)
    data = data.drop(data.index[tmp]).reset_index()
    return data.drop('index', axis=1)
    
def Clearing(data):
    years_c = [item for ind, item in enumerate(np.array(data.columns)) if ind not in (range(0,4))]
    if data['Country Code'].get_value(len(data)-5) == np.nan:
        data=data.drop(data.index[[range(len(data)-5, len(data))]]).reset_index()
    data = Dropempt(Fillnan(data, years_c), years_c)
    return data


# Сортировка по регионам.

# In[5]:


def Make_region(code, reg='region'):
    #print code
    if code in data_cnt['name'].values:
        #print code
        name = (data_cnt['name'][data_cnt['name'] == code].index.tolist())[0]
        return data_cnt[reg].get_value(name)
    else:
        return np.nan

def Sorting(data, reg='region'):
    #if data['Country Name'].get_value(len(data)-5) == np.nan:
    #    data=data.drop(data.index[[range(len(data)-5, len(data))]]).reset_index()
    if 'Region' not in data.columns:
        data['Region'] = data['Country Name'].apply(lambda x: Make_region(x, reg))
        data['Region'].dropna()
        return data.reset_index()
    else:
        return data


# In[6]:


data1 = Clearing(data1)
#заметил много шума.
del data1['2016 [YR2016]']

data2 = Clearing(data2)
del data2['2001 [YR2001]']

data3 = Clearing(data3)
del data3['2016 [YR2016]']


# Эвристика
# Преимущество данного подхода в том, что мы не потеряем единственные значения у некоторых стран. Но при этом, отразим актуальность, а значит и реальность.

# In[7]:


def data_inc(data):
    magic = 0.08
    for ind, item in enumerate(np.array(data.columns)):
        if ind not in (range(0,4)):
        #попробую с интерполяцией
            data[item] = data[item].apply(lambda x: ((ind-2)*magic)*x if ((ind-2)*magic)>1 else x).interpolate()
    return data

def R_mean(data):
    if ('Mean' not in data.columns) and ('Region' not in data.columns):
        data['Mean']= data[[item for ind, item in enumerate(np.array(data.columns)) if ind not in (range(0,4))]].apply(lambda x: np.nanmean(x), axis = 1)
    elif ('Region' in data.columns):
        data['Mean']= data[[item for ind, item in enumerate(np.array(data.columns)) if ind not in [(range(0,4)), len(data.columns)-1]]].apply(lambda x: np.nanmean(x), axis = 1)
    return data


# Чувствую себя отвратительно за такой код. Доделаю до конца и поправлю. 

# In[8]:


#data1 = R_mean(data_inc(data1))
#data2 = R_mean(data_inc(data2))
#data3 = R_mean(data_inc(data3))


# Вот тут буду коррелировать, строить графики и пр

# In[9]:


def podshape(t1,t2):
    cntrs1 = t1['Country Name'].tolist()
    cntrs2 = t2['Country Name'].tolist()
    x, y = [],[]
    if len(t1)!=len(t2):
        for contr in cntrs1:
            if contr in cntrs2:
                x.append(t1['Mean'][t1['Country Name']== contr].as_matrix()[0])
                y.append(t2['Mean'][t2['Country Name']== contr].as_matrix()[0])
        return x, y
    else:
        return t1['Mean'].as_matrix(), t2['Mean'].as_matrix()

def Shaping(t1,t2): 
    if len(t1)<len(t2):
        return podshape(t1,t2)
    elif len(t1)>len(t2):
        return podshape(t1,t2)
    elif len(t1)==0 or len(t1)==0:
        return True
    else:
        return t1['Mean'].as_matrix(), t2['Mean'].as_matrix()
    
def Correlation(data_r,regions=[], where='region', name='Damn'):
    data = data_r.copy()
    if where == 'region':
        data = Sorting(data)
    else:
        data = Sorting(data, where)
        
    if len(regions)==0:
        regions = data['Region'].dropna().unique()
    
    codes = data['Series Code'].dropna().unique()
    rez=pd.DataFrame()    
    
    tmp = []
    
    for region in regions:
        for code1 in codes:
            for code2 in codes:
                tmp = []
                Sh = Shaping(data[['Mean', 'Country Name']][data['Series Code']==code1][data['Region']==region], data[['Mean', 'Country Name']][data['Series Code']==code2][data['Region']==region])
                if Sh!=True:
                    tmp = sts.pearsonr(Sh[0],Sh[1])
                    
                if len(tmp)!=0 and np.abs(tmp[0])>0.099:
                    rez = rez.append(pd.DataFrame(tmp[0], columns=[code1+':'+region], index=[code2+':'+region+':'+'cor-value']))
                    rez = rez.append(pd.DataFrame(tmp[1], columns=[code1+':'+region], index=[code2+':'+region+':'+'p-value']))
                    rez = rez.groupby(rez.index).first()
        
        filename = name+'.Corr_in_'+str(region)+'.xlsx'
        rez.to_excel(filename, encoding='utf-8')
        rez = pd.DataFrame()
        print (str(region))


# In[10]:


#data_cnt['sub-region'].unique()


# Correlation(data2, where='region', name='data2')
# 
# Correlation(data2, where='sub-region', name='data2')

# Correlation(data3, where='region', name='data3')
# 
# Correlation(data3, where='sub-region', name='data3')

# Correlation(data1, where='region', name='data1')
# 
# Correlation(data1, regions =[u'Western Europe', u'Eastern Europe',
#        u'Central America', u'Western Africa', u'Northern America',
#        u'Southern Africa', u'South-Eastern Asia', u'Eastern Africa',
#        u'Eastern Asia', u'Melanesia', u'Micronesia', u'Central Asia'], where='sub-region', name='data1')

# Общая таблица

# In[11]:


all_data = pd.DataFrame().append(data1).append(data2).append(data2)


# In[12]:


years_c = [item for ind, item in enumerate(np.array(data1.columns)) if ind not in (range(0,4)) and item !='Mean']


# Correlation(all_data, where='region', name='all')
# 
# Correlation(all_data, regions=[u'Western Africa', u'Northern America',
#        u'Southern Africa', u'South-Eastern Asia', u'Eastern Africa',
#        u'Eastern Asia', u'Melanesia', u'Micronesia', u'Central Asia'], where='sub-region', name='all')

# Какое же это фиаско
# 
# Продолжаем кодить, на этот раз смотрим по стране без среднего.

# In[58]:


def check_vec(X, Y, years_c, procent=1):
    if procent!=1:
        for i in range(len(X)):
            if (X[i]==np.nan or Y[i]==np.nan) and (len(X)>=len(years_c)*procent or len(Y)>=len(years_c)*procent):
                del X[i]
                del Y[i]
        return X,Y
    
    elif (np.nan not in X and np.nan not in Y):
        return X,Y
    
    else:
        return [],[]
    
def file_making(chlst, dir_name):
    for name in chlst:
        #print (name, os.listdir(dir_name))
        if name in os.listdir(dir_name):
            return True
            

def C_corr(data_r, country='RUS',procent=100, reg=True, dir_name = 'Correlations'):
    
    #вход в функцию и начало отсчета.
    logging.debug('Starting')
    start_time = datetime.datetime.now()
    
    #копируем, и определяем результрующий датафрейм.
    data = data_r.copy()
    rez=pd.DataFrame() 
    
    #записываем года и коды из датасета и регион
    years_c = [item for ind, item in enumerate(np.array(data1.columns)) if ind not in (range(0,4)) and item !='Mean' and item!='Region']
    codes = data['Series Code'].dropna().unique()
    region = Make_region(data[data['Country Code']==country]['Country Name'].as_matrix()[0], 'sub-region')
    
    filenames = [('deep'+'.Corr_in_'+str(country)+'.xlsx'),('deep'+'.Corr_in_'+str(country)+'.'+str(region)+'.xlsx'),('deep'+'.Corr_in_'+str(country)+'NONE.xlsx')]
    
    #делим проценты и опр. счетчик операций.
    procent = procent/100
    k=0
    
    #проверка директории
    if dir_name not in os.listdir():
        os.mkdir(dir_name)
    
    #проверка на наличие файла.
    if file_making(filenames, dir_name)==True:
        logging.debug('File exist!')
        return 0
    
    for ind , code1 in enumerate(codes):
        for jnd, code2 in enumerate(codes):
            k+=1
            
            #идем ниже диагонали
            if ind>jnd:
                X = data[(data['Country Code']==country) & (data['Series Code']==code1)][years_c].as_matrix()
                Y = data[(data['Country Code']==country) & (data['Series Code']==code2)][years_c].as_matrix()
                
                logging.debug(str(round((k*100)/(len(codes)**2),2))+'% in '+country)
                
                #проверяем наличие данных в двух векторах.
                if len(X)!=0 and len(Y)!=0:
                    
                    #исходя из процентов выбрасываем нан, либо возвращаем пустые массивы.
                    X,Y = check_vec(X[0],Y[0], years_c, procent)
                    
                    #проверяем чтобы оклонение было хорошее(чтобы не получилось, что данные лежат в одной точке, тогда корреляции не получается.)
                    #заодно чекаем пустоту массивов.
                    if (X.std()>0.7 and Y.std()>0.7) and (len(X)!=0 and len(Y)!=0):
                        #корреляция пирсона
                        tmp = sts.pearsonr(X,Y)
                        
                        #проверяем на пустоту корреляцию(малоли), а также, отсекаем малленькие корреляции.
                        if len(tmp)!=0 and np.abs(tmp[0])>0.099:
                            
                            #Заполняем таблицу, (тут оптимизировать.)
                            rez = rez.append(pd.DataFrame(tmp[0], columns=[code1+':'+country], index=[code2+':'+country+':'+'cor-value']))
                            rez = rez.append(pd.DataFrame(tmp[1], columns=[code1+':'+country], index=[code2+':'+country+':'+'p-value']))
                            rez = rez.groupby(rez.index).first()
    
                     
    
    #нужно ли указывать регион или сабрегион
    if reg==False:
        rez.to_excel(dir_name + '/' + filenames[0], encoding='utf-8')
    elif region!=np.nan:
        rez.to_excel(dir_name + '/' + filenames[1], encoding='utf-8')
    else:
        rez.to_excel(dir_name + '/' + filenames[2], encoding='utf-8')
    
    #время рассчета выводим и записываем в лог.
    print ('Time elapsed:', datetime.datetime.now() - start_time)
    print (country)
    
    currentDay = datetime.datetime.now().day
    currentMonth = datetime.datetime.now().month
    
    f = open((str(currentDay)+'.'+str(currentMonth)+'.txt'),'w')
    f.write(('\n'+'Time elapsed:'+str(datetime.datetime.now() - start_time)+' in '+country + '\n Time now:'+datetime.datetime.now()))
    f.close()
    
    logging.debug('Exiting')


# In[57]:


#C_corr(all_data)


# In[ ]:


if __name__== '__main__':
    pool = Pool(processes=int(sys.argv[1]))
    func = partial(C_corr, all_data, procent=100, reg=True)
    pool.map(func, all_data['Country Code'].unique())
    pool.close()
    pool.join()


# tmp1, tmp2 = all_data[(all_data['Series Name'] == 'Improved sanitation facilities (% of population with access)') & (all_data['Country Code']=='RUS')][years_c].as_matrix()[0], all_data[(all_data['Series Code'] == 'SI.POV.NAHC') & (all_data['Country Code']=='RUS')][years_c].as_matrix()[0]

# Make_region('Angola')
