
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
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s] %(message)s',
                    )


# In[2]:


def fillnan(data, years_c):
    for name in years_c:
        #заменяем троеточия
        data[name] = data[name].apply(lambda x: np.nan if x==('..') else float(x))

    return data

def dropempt(data, years_c):
    data = fillnan(data, years_c)
    return data.dropna(thresh=8)

def Clearing(data):
    years_c = [item for ind, item in enumerate(np.array(data.columns)) if item not in ['Country Name', 'Country Code','Series Code','Series Name']]
    print ('Before clear:', len(data))

    data = dropempt(data, years_c)

    print ('After clear:', len(data))
    return data


# In[3]:


def Make_region(code, reg='region'):
    #print code
    if code in data_cnt['name'].values:
        #print code
        name = (data_cnt['name'][data_cnt['name'] == code].index.tolist())[0]
        return data_cnt[reg].at[name]
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


# In[7]:


def check_vec(X, Y, years_c, procent=1):
    if procent!=1:
        for i in range(len(X)):
            if (X[i]==np.nan or Y[i]==np.nan) and ((len(X)>=len(years_c)*procent and len(Y)>=len(years_c)*procent)):
                del X[i]
                del Y[i]
        return np.array(X),np.array(Y)

    elif (np.nan not in X and np.nan not in Y):
        return np.array(X),np.array(Y)

    else:
        return [],[]

def file_making(chlst, dir_name):
    for name in chlst:
        #print (name, os.listdir(dir_name))
        if name in os.listdir(dir_name):
            return True

def C_corr(data_r, country='RUS', procent=100, reg=True, dir_name = 'Correlations'):

    #вход в функцию и начало отсчета.
    logging.debug('Starting')
    start_time = datetime.datetime.now()

    #копируем, и определяем результрующий датафрейм.
    data = data_r.loc[data_r['Country Code'] == country]
    del data_r
    rez=pd.DataFrame()

    #записываем года и коды из датасета и регион
    years_c = [item for ind, item in enumerate(np.array(data.columns)) if item not in ['Country Name', 'Country Code','Series Code','Series Name'] and item !='Mean' and item!='Region']
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
        if reg==False:
            Sumup(dir_name, filenames[0])
        elif region!=np.nan:
            Sumup(dir_name, filenames[1])
        else:
            Sumup(dir_name, filenames[2])
        return 0

    #logging.debug(country+':')
    for ind , code1 in tqdm(enumerate(codes), desc = country, total = len(codes)):
        for jnd, code2 in enumerate(codes):
            k+=1

            #идем ниже диагонали
            if ind>jnd:
                X = data.loc[(data['Country Code']==country) & (data['Series Code']==code1)][years_c].as_matrix()
                Y = data.loc[(data['Country Code']==country) & (data['Series Code']==code2)][years_c].as_matrix()

                #logging.debug(str(round((k*100)/(len(codes)**2),2))+'% in '+country)

                #проверяем наличие данных в двух векторах.
                if len(X)!=0 and len(Y)!=0:

                    #исходя из процентов выбрасываем нан, либо возвращаем пустые массивы.
                    X,Y = check_vec(X[0],Y[0], years_c, procent)

                    #проверяем чтобы оклонение было хорошее(чтобы не получилось, что данные лежат в одной точке, тогда корреляции не получается.)
                    #заодно чекаем пустоту массивов.
                    if (len(X)!=0 and len(Y)!=0) and (np.std(X)>0.7 and np.std(Y)>0.7):
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
        Sumup(dir_name, filenames[0])
    elif region!=np.nan:
        rez.to_excel(dir_name + '/' + filenames[1], encoding='utf-8')
        Sumup(dir_name, filenames[1])
    else:
        rez.to_excel(dir_name + '/' + filenames[2], encoding='utf-8')
        Sumup(dir_name, filenames[2])

    #время рассчета выводим и записываем в лог.
    print ('Time elapsed:', datetime.datetime.now() - start_time)
    print (country)

    currentDay = datetime.datetime.now().day
    currentMonth = datetime.datetime.now().month
    currentYear = datetime.datetime.now().year

    f = open((str(currentDay)+'.'+str(currentMonth)+'.'+str(currentYear)+'.txt'), 'a+')
    log = f.read()+(('\n'+'Time elapsed:'+str(datetime.datetime.now() - start_time)+' in '+country + '\n Time now:'+str(datetime.datetime.now())))
    f.write(log)
    f.close()

    logging.debug('Exiting')


# In[26]:


def D_maker(data,c=0.5, p=0.01, m=1, s=u'Корреляция'):
    rez=[]
    for index in data.index:
        for column in data.columns:
            i_tmp = index.split(':')
            c_tmp = column.split(':')
            value=[]

            if i_tmp[2]!=u'p-value' and i_tmp[0]!=c_tmp[0]:
                value.append([data.get_value(index=index, col=column),data.get_value(index=i_tmp[0]+':'+i_tmp[1]+':p-value', col=column)])

                if abs(value[0][0])>c and value[0][1]<p and abs(value[0][0])<m:

                    if [defen[defen['Code']==c_tmp[0]]['Indicator Name'].tolist()[0], defen[defen['Code']==i_tmp[0]]['Indicator Name'].tolist()[0], value[0][0]] not in rez:
                        rez.append([defen[defen['Code']==i_tmp[0]]['Indicator Name'].tolist()[0], defen[defen['Code']==c_tmp[0]]['Indicator Name'].tolist()[0], value[0][0]])

                    #Degbug
                    #return rez
    if len(rez) == 0:
        rez = [['EMTY CELL', "EMPTY CELL", 696]]
    #print(rez)
    db = pd.DataFrame().append(rez)
    #print (db)
    db.columns = [u'Первый признак', u'Второй признак', u'Корреляция']
    db = check(db.sort_values(by=[s]))

    #print (len(db))
    return db

def check(data):
    rez=[]
    data = data.sort_values(by=[u'Корреляция'])
    tmp = data.as_matrix()
    for i, row in enumerate(tmp):
        if i<len(tmp)-1:
            if row[2] != tmp[i+1][2] and row[1] != tmp[i+1][0] and row[0] != tmp[i+1][1]:
                rez.append(row)
    if len(rez)!=0:
        db = pd.DataFrame().append(rez)
        db.columns = [u'Первый признак', u'Второй признак', u'Корреляция']
        return db.reset_index()
    else:
        print('what?')
        return data


def to_ex(data, filename):
    data[[u'Первый признак', u'Второй признак', u'Корреляция']].sort_values(by=u'Первый признак').to_excel(filename, sheet_name=filename.split('.')[1], index = False)

def Sumup(d_name, name):
    #проверка директории
    if 'Conclusion' not in os.listdir():
        os.mkdir('Conclusion')

    rez = pd.DataFrame(columns = [u'Первый признак', u'Второй признак', u'Корреляция'])
    n_count = name.split('.')[1].split('_')[2]
    filename = name.split('.')[2]+'.'+n_count+'.Con.xlsx'
    print (n_count)
    if filename not in os.listdir('Conclusion'):
        print('inside!', d_name+'/'+filename)
        data = pd.read_excel(d_name+'/'+name ,encoding = 'utf8')
        if len(data)>4:
            to_ex(D_maker(data), 'Conclusion'+'/'+filename)


# In[ ]:


if __name__== '__main__':
    df = pd.read_excel('Data_Extract_From_Gender_Statistics.xlsx',
                         encoding = 'utf8').append(pd.read_excel('Data_Extract_From_Health_Nutrition_and_Population_Statistics.xlsx',
                                                                                                                  encoding='utf8')).append(pd.read_excel('Data_Extract_From_Millennium_Development_Goals.xlsx',
                                                                                                                  encoding='utf8')).append(pd.read_excel('Data_Extract_From_Health_Nutrition_and_Population_Statistics_by_Wealth_Quintile.xlsx',
                                                                                                                  encoding='utf8'))
    data_cnt = pd.read_csv('all.csv', encoding='utf8')
    df = Clearing(df)
    defen = pd.read_excel('Data_Extract_From_Gender_Statistics.xlsx', sheet_name=1, encoding = 'utf8').append(pd.read_excel('Data_Extract_From_Health_Nutrition_and_Population_Statistics.xlsx', sheet_name=1, encoding='utf8')).append(pd.read_excel('Data_Extract_From_Millennium_Development_Goals.xlsx', sheet_name=1, encoding='utf8'))

    pool = Pool(processes=11)
    func = partial(C_corr, df, procent=80, reg=True)
    pool.map(func, df['Country Code'].unique())
    pool.close()
    pool.join()

