import numpy as np 
import pandas as pd
import model_gp as gp
import model_arima as ar 
import model_lstm as lstm 
from mosqlient.scoring import Scorer

import sys 
sys.path.append('../')
from methods.ensemble import Ensemble

def train_models(state, start_train_date, end_train_date): 

    print(state)
    
    df_org = pd.read_csv(f'data/dengue_{state}.csv.gz')
    df_org['date'] = pd.to_datetime(df_org['date'])

    # input to arima model
    print('--------------------- Training ARIMA ---------------------')

    ar.train_model(df_org[['date', 'casos']], state, train_ini_date = start_train_date, train_end_date = end_train_date)

    # train gpr model 
    print('--------------------- Training GP ---------------------')

    gp.train_model(state, ini_train = start_train_date, end_train = end_train_date)

    # train lstm model
    print('--------------------- Training LSTM ---------------------')


    feat = 6
    HIDDEN = 64
    LOOK_BACK = 4
    PREDICT_N = 3

    model = lstm.build_lstm(hidden=HIDDEN, features=feat, predict_n=PREDICT_N, look_back=LOOK_BACK,
                                batch_size=4, loss='mse')

    model.compile(loss='mse', optimizer='adam', metrics=["accuracy", "mape", "mse"])
            
    lstm.train_model(model, state, doenca='dengue',
                        end_train_date=None,
                        ratio = 1,
                        ini_date = start_train_date,
                        end_date = end_train_date,
                        filename=f'data/dengue_{state}.csv.gz',
                        min_delta=0.001, label='state',
                        patience = 30, 
                        epochs=300,
                        batch_size=4,
                        predict_n=PREDICT_N,
                        look_back=LOOK_BACK)


def apply_models(state, end_date): 
    '''
    load the trained models, apply then and concatenate predictions
    '''
    df_arima = ar.apply_model(state, end_date)
    
    df_gp = gp.apply_model(state, end_date)
                
    FILENAME_DATA = f'data/dengue_{state}.csv.gz'
    df_ = pd.read_csv(FILENAME_DATA, index_col = 'date')
    
    feat = df_.shape[1]
            
    model_name = f'trained_{state}_dengue_state'

    
    df_lstm = lstm.apply_forecast(state, None, end_date, look_back=4, predict_n=3,
                                            filename=FILENAME_DATA, model_name=model_name)
    
    df_concat = pd.concat([df_arima, df_gp, df_lstm])

    return df_concat

def format_pred(df, label_epiweek, step= None): 
    '''
    Function to filter and rename the columns of the prediction dataframe
    '''
    
    if step == None: 
        df = df.loc[(df.epiweek == label_epiweek)]

    else: 
        df = df.loc[(df.epiweek == label_epiweek) & (df.step == step)]
    
    df = df.rename(columns = {'model':'model_id'})

    df['model_id'] = df['model_id'].replace({'arima':1, 'gp':2, 'lstm':3 })

    df = df.reset_index(drop = True)

    df.date = pd.to_datetime(df.date)
    
    return df[['date', 'pred', 'lower_95', 'upper_95', 'model_id']]


def get_weights_crps(CRPSm1, CRPSm2, CRPSm3): 
    '''
    compute the weights based in the CRPS
    '''
    w1 = (1/(CRPSm1) )/(1/(CRPSm1) + 1/(CRPSm2) + 1/(CRPSm3)) 
    w2 = (1/(CRPSm2) )/(1/(CRPSm1) + 1/(CRPSm2) + 1/(CRPSm3)) 
    w3 = (1/(CRPSm3) )/(1/(CRPSm1) + 1/(CRPSm2) + 1/(CRPSm3)) 

    return [w1, w2, w3]

def get_weights(api_key, df_preds, casos): 

    ens_log = Ensemble(df = df_preds,
        order_models = [1, 2, 3], 
        dist = 'log_normal',
        mixture = 'log',
        fn_loss = 'median', 
             conf_level=0.95)

    weights_log = ens_log.compute_weights(casos, metric = 'crps')['weights']
    
    ens_lin = Ensemble(df = df_preds,
            order_models = [1, 2, 3], 
            dist = 'log_normal',
            mixture = 'linear',
            fn_loss = 'median', 
                 conf_level=0.95)
    
    weights_lin = ens_lin.compute_weights(casos, metric = 'crps')['weights']

    crps_models = list()
    for model_id in [1,2,3]:
        
        score_model = Scorer(api_key,
                      df_true = casos,
                      pred=df_preds.loc[df_preds.model_id == model_id],
                      dist='log_normal',
                      fn_loss='median',
                      conf_level=0.95)
    
        crps_models.append(score_model.crps[0]['pred'].values[0])
    
    weights_crps = get_weights_crps(crps_models[0], crps_models[1], crps_models[2])
    
    return weights_lin, weights_log, np.array(weights_crps)

def apply_ensemble(df_preds, weights_lin, weights_log, weights_crps): 

    len_models = len(weights_lin)
    weights_equal =  np.ones(len_models)/len_models

    ens_log = Ensemble(df = df_preds,
        order_models = [1, 2, 3], 
        dist = 'log_normal',
        mixture = 'log',
        fn_loss = 'median', 
             conf_level=0.95)
  
    ens_lin = Ensemble(df = df_preds,
            order_models = [1, 2, 3], 
            dist = 'log_normal',
            mixture = 'linear',
            fn_loss = 'median', 
                 conf_level=0.95)
    
    df_ens_lin = ens_lin.apply_ensemble(weights = weights_lin, p = [0.025, 0.5, 0.975])
    
    df_ens_lin_equal = ens_lin.apply_ensemble(weights = weights_equal, p = [0.025, 0.5, 0.975])

    df_ens_log = ens_log.apply_ensemble(weights = weights_log, p = [0.025, 0.5, 0.975])
    
    df_ens_log_equal = ens_log.apply_ensemble(weights = weights_equal, p = [0.025, 0.5, 0.975])

    df_ens_log_crps = ens_log.apply_ensemble(weights = weights_crps, p = [0.025, 0.5, 0.975])
    
    df_ens_lin_crps = ens_lin.apply_ensemble(weights = weights_crps, p = [0.025, 0.5, 0.975])

    for df_ in [df_ens_lin, df_ens_lin_equal, df_ens_lin_crps, df_ens_log, df_ens_log_equal, df_ens_log_crps]: 

        df_['step'] = [1,2,3]

    return df_ens_lin, df_ens_lin_equal,  df_ens_lin_crps, df_ens_log, df_ens_log_equal, df_ens_log_crps 