import joblib
import pickle
import pandas as pd 
from model_gp import get_next_n_weeks
from pmdarima import preprocessing as ppc
from mosqlient.forecast import Arima
from scipy.special import inv_boxcox


def get_prediction_dataframe(model, date, boxcox, horizon) -> pd.DataFrame:
    """
    Function to organize the predictions of the ARIMA model in a pandas DataFrame.

    Parameters
    ----------
    horizon: int
        The number of weeks forecasted by the model
    end_date: str
        Last week of the out of the sample evaluation. The first week is after the last training observation.
    plot: bool
        If true the plot of the model out of the sample is returned
    """

    dfs = []
    for alpha in [0.05, 0.1, 0.2, 0.5]:
        if horizon is None:
            preds_ = model.predict_in_sample(return_conf_int=True, alpha=alpha)
        else:
            preds_ = model.predict(
                n_periods=horizon, return_conf_int=True, alpha=alpha
            )
        df_ = pd.DataFrame(
            preds_[1],
            columns=[
                f"lower_{int((1-alpha)*100)}",
                f"upper_{int((1-alpha)*100)}",
            ],
        )
        dfs.append(df_)

    df_preds = pd.concat(dfs, axis=1)

    try:
        df_preds["pred"] = preds_[0].values
    except:
        df_preds["pred"] = preds_[0]

    for col in df_preds.columns:
        df_preds[col] = boxcox.inverse_transform(df_preds[col])[0]
    
    df_preds["date"] = date

    return df_preds

def train_model(df, state, train_end_date, train_ini_date = '2015-01-01'):
    '''
    Function to train and save the arima model 
    '''
    df = df.rename(columns = {'date': 'dates',
                            'casos': 'y'})

    df['y'] = inv_boxcox(df['y'].values, 0.05) - 1

    df_ = df.copy()

    df_.dates = pd.to_datetime(df_.dates)

    df_.set_index('dates', inplace = True)
    
    m_arima = Arima(df = df_)

    model = m_arima.train( train_ini_date=train_ini_date, train_end_date = train_end_date)

    # Save model
    with open(f'saved_models/arima_{state}.pkl', 'wb') as pkl:
        pickle.dump(model, pkl)
    
    # save transf on data
    bc_transformer = m_arima.boxcox
    joblib.dump(bc_transformer, f'saved_models/bc_{state}.pkl')

def apply_model(state, end_date):
    '''
    Function to load and apply the pre trained model 
    '''
    
    df_ = pd.read_csv(f'data/dengue_{state}.csv.gz', usecols = ['date', 'casos'])

    df_ = df_.loc[df_.date <= end_date]
    
    df_ = df_.rename(columns = {'date': 'dates',
                                    'casos': 'y'})

    df_['y'] = inv_boxcox(df_['y'].values, 0.05) - 1

    df_.set_index('dates', inplace = True)

    df_.index = pd.to_datetime(df_.index)

    bc = joblib.load(f'saved_models/bc_{state}.pkl')

    df_.loc[:, "y"] = bc.transform(df_.y)[0]

    with open(f'saved_models/arima_{state}.pkl', 'rb') as pkl:
        m_arima = pickle.load(pkl)

    # update the model with the new data:
    m_arima.update(df_)

    date = get_next_n_weeks(df_.index[-1].strftime("%Y-%m-%d"), 3)

    df_for = get_prediction_dataframe(m_arima, date, bc, 3)

    df_for['step'] = [1,2,3]

    df_for['model'] = 'arima'

    #for col in df_for.columns: 
    #    if col != 'date':
    #        df_for[col] = inv_boxcox(df_for[col].values, 0.05) - 1

    #df_for.to_csv(f'forecast_tables/for_arima_se_{for_week}_{state}.csv.gz', index = False)

    return df_for  

