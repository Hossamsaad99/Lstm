

import numpy as np
import pandas_datareader as pdr
from fastapi import FastAPI
import uvicorn
import pydantic
import gunicorn
import httptools
from datetime import *
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


    

@app.get('/')

def index():
    return {'message': 'Hello!'}

@app.post('/predict')

async def predict_price(data: str):
    if data == 'F':
      
      model_prediction, lstm_prediction = lstm(data)
      

      return {
        
        'LSTM prediction' : lstm_prediction
        
            }

    else:
      return {"the ticker not supported yet"}

        
      
       
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
    

