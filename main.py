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

app = FastAPI()

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
    
    fc ,diff = Arima(ticker)
    return {'prediction':fc[0],'difference':diff[0]}
       
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
    

