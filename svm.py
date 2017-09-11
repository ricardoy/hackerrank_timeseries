import numpy as np
import pandas as pd
import random
import sys
from datetime import timedelta
from datetime import datetime
from pandas import DataFrame
from sklearn.svm import SVR

initial_date = datetime.strptime('2012-10-01', '%Y-%m-%d')

series = []
sys.stdin.readline()
for line in sys.stdin:
    series.append(line.strip())
series = np.array(series, dtype=np.float)


def dataframe_add_shift(df, t):
    key = 'shift_%d' % (t)
    df[key] = df.shift(t)['user_sessions']
    return df


def get_X_y(df):
    '''
    Somente linhas com todos os valores válidos serão devolvidas. As 
    features utilizadas no treinamento são: mês, ano, dia do mês, dia
    do ano e representação one-hot (https://en.wikipedia.org/wiki/One-hot)
    dos 7 dias da semana.
    '''
    df = df.dropna()
    df = df.drop(['date', 'weekday'], axis=1)
    X = df.drop(['user_sessions'], axis=1)
    y = df['user_sessions']
    return X, y


def generate_dataframe(initial_date, series, window_size=0):
    '''
    Gera as features de cada dia, associados com o número de sessões
    de usuários dos últimos 30 dias e, partindo de 35 dias, de 7 em 7 dias
    até completar 29 semanas.
    '''
    series = np.concatenate((series, np.zeros(30)))
    df = DataFrame(series, 
                   columns=['user_sessions'],
                   )
    
    df['user_sessions'] = df['user_sessions'] / 1000

    df['date'] = pd.date_range(initial_date, periods=len(series), freq='D')
    df['weekday'] = df['date'].dt.weekday_name
    df['year'] = df['date'].dt.year - 2012
    df['month'] = df['date'].dt.month / 12.
    df['day'] = df['date'].dt.day / 31.
    df['dayofyear'] = df['date'].dt.dayofyear / 366.
    df = df.join(pd.get_dummies(df['weekday']))    
    
    for i in range(1, 30):
        df = dataframe_add_shift(df, i)
        
    for i in range(5, 30):
        df = dataframe_add_shift(df, i * 7)

    return df
    
    
df = generate_dataframe(initial_date, series)

X, y = get_X_y(df)

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.05)

'''
X e y já contêm as linhas dos 30 dias para os quais deve ser
calculado o número de sessões, por isso o slice final em -30.
Os dados indexados por [-90:-30] representam os últimos 60 
dias dos dados de entrada.
'''
svr_rbf.fit(X[-90:-30], y[-90:-30])

offset = len(series)
for i in range(-30, 0):
    if i != -1:
        r = svr_rbf.predict(X[i:i+1])
    else:
        r = svr_rbf.predict(X[-1:])
    df.set_value(offset + 30 + i, 'user_sessions', r)
        
    for i in range(1, 30):
        df = dataframe_add_shift(df, i)
        
    for i in range(5, 30):
        df = dataframe_add_shift(df, i * 7)    
        
    X, y = get_X_y(df)
    
for i in range(-30, 0):
    print(int(df.iloc[i]['user_sessions'] * 1000))
