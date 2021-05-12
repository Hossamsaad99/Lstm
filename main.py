#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pandas_datareader as pdr
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
from fastapi import FastAPI
import uvicorn
import pydantic
import gunicorn
import httptools
from tensorflow import keras
from keras.models import load_model
app = FastAPI()

def lstm(data_set):
  """
  Getting the desired data from yahoo, then doing some data manipulation such as data
  reshaping
  Args:
      (str) data_set - the ticker of desired dataset (company)
  Returns:
      (float) diff_prediction - the model out-put (the prediction of the next day)
      (float) real_prediction - the model output + today's price (real price of tomorrow)
  """

  # data gathering
  df = pdr.DataReader(data_set, data_source='yahoo', start=date.today() - timedelta(100))

  # data manipulation

  # creating a new df with Xt - Xt-1 values of the close prices (most recent 60 days)
  close_df = df['2012-01-01':].reset_index()['Close'][-61:]
  close_diff = close_df.diff().dropna()
  data = np.array(close_diff).reshape(-1, 1)

  # reshaping the data to 3D to be accepted by our LSTM model
  model_input = np.reshape(data, (1, 60, 1))

  # loading the model and predicting
  loaded_model = load_model("lstm_f_60.hdf5")
  model_prediction = float(loaded_model.predict(model_input))
  real_prediction = model_prediction + df['Close'][-1]
  

  return model_prediction, real_prediction

def Regression(ticker):
  """
  Forcasting using an ensambled model between SVR, Ridge and Linear regression! by Getting the desired data from yahoo, 
  then doing some data manipulation
  Args:
      (str) ticket - the ticker of desired dataset (company)
  Returns:
      (float) arima_output - the model out-put (the prediction of the next day)
      (float) diff - the model output - today's price (the diff between tomorrow's prediction and today's real value)
  """
  start_date = datetime.now() - timedelta(1)
  start_date = datetime.strftime(start_date, '%Y-%m-%d')

  df = pdr.DataReader(ticker, data_source='yahoo', start=start_date)  # read data
  df.drop('Volume', axis='columns', inplace=True)
  X = df[['High', 'Low', 'Open', 'Adj Close']]  # input columns
  y = df[['Close']]  # output column
  input = X
  loaded_model = pickle.load(open('regression_model.pkl', 'rb'))
  reg_prediction = loaded_model.predict(input)
  reg_diff=reg_prediction-df.Close[-1]

  return  reg_prediction,reg_diff

def Arima(ticker):
    
    #Dataset of Ticker
    df = pdr.DataReader(ticker, data_source='yahoo', start='2016-01-01')
    df.index = pd.to_datetime(df.index, format="%Y/%m/%d")
    df = pd.Series(df['Close'])
    last_day=df[-1]
    #Best Order
    auto_order = pm.auto_arima(df, start_p=0, start_q=0, test='adf', max_p=3, max_q=3, m=1,d=None,seasonal=False   
                      ,start_P=0,D=0, trace=True,error_action='ignore',suppress_warnings=True,stepwise=True)
    best_order = auto_order.order
    # Fit Model
    model = ARIMA(df, order=best_order)
    model_fit = model.fit(disp=0)
    fc ,se, conf = model_fit.forecast(1)
    
    diff = fc - last_day
    
    return fc , diff
    

@app.get('/')

def index():
    return {'message': 'Hello!'}

@app.post('/predict')

async def predict_price(ticker:str):
    if data == 'F':
      
      arima_prediction, diff = arima(data)
      model_prediction, lstm_prediction = lstm(data)
      reg_prediction,reg_diff = Regression(data)
  
    return {'Arima prediction' : arima_prediction[0],'LSTM prediction' : lstm_prediction,regression prediction' : reg_prediction[0]}
        
        '
       
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
    

