import numpy as np
import pandas as pd
from epiweeks import Week
from scipy.stats import boxcox

def filter_agg_data(state, column = 'casos_est'):
    '''
    Get the state data and aggregate it by week
    '''
    df = pd.read_parquet(f'data/{state}_dengue.parquet', columns=['municipio_geocodigo', 'casos_est'])

    df.index = pd.to_datetime(df.index) 
        
    df = df.resample('W-SUN').sum().drop(['municipio_geocodigo'], axis =1).reset_index()

    df = df.rename(columns = {column: 'casos', 'data_iniSE':'date'})

    df['uf'] = state

    return df 

# funções para criar as features utilizadas para treinar os modelos de forecast
def calcular_metricas_por_janela(array, tamanho_janela, funcoes):
    # Criar um array com as janelas deslizantes
    janelas = np.lib.stride_tricks.sliding_window_view(array, tamanho_janela)

    # Aplicar as funções de interesse em cada janela
    resultados = [func(janela, axis=0) for func in funcoes for janela in janelas]
    
    return np.array(resultados)

def get_slope(casos, axis = 0): 
     
    return np.polyfit(np.arange(0,4), casos, 1)[0]

def org_data(df):
    df = df.copy()
    df.date = pd.to_datetime(df.date)
    
    df.set_index('date', inplace = True)

    df = df[['casos']].resample('W-SUN').sum()

    df['casos'] = boxcox(df.casos+1, lmbda=0.05)

    df['SE'] = [Week.fromdate(x) for x in df.index]

    df = df[['SE', 'casos']].sort_index()
    
    df['SE'] = df['SE'].astype(str).str[-2:].astype(int)
    
    df['SE'] = df['SE'].replace(53,52)
    
    df['diff_casos'] = np.concatenate( (np.array([np.nan]), np.diff(df['casos'], 1)))
    
    array = np.array(df.casos)
    tamanho_janela = 4
    
    df['casos_mean'] =  np.concatenate( (np.array([np.nan, np.nan, np.nan]), calcular_metricas_por_janela(array, tamanho_janela, [np.mean])))
    
    df['casos_std'] =  np.concatenate( (np.array([np.nan, np.nan, np.nan]), calcular_metricas_por_janela(array, tamanho_janela, [np.std])))
    
    df['casos_slope'] =  np.concatenate( (np.array([np.nan, np.nan, np.nan]), calcular_metricas_por_janela(array, tamanho_janela, [get_slope])))
    
    df = df.dropna()

    return df 