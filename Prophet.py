from prophet import Prophet
import logging
import warnings
warnings.filterwarnings('ignore')
from IPython.display import clear_output 
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

%matplotlib inline

# Function for Evaluate the model
def make_comparison_dataframe(historical, forecast):
    return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical.set_index('ds'))


def calculate_forecast_errors(df, prediction_size):
    
    df = df.copy()
    
    df['e'] = df['y'] - df['yhat']
    df['p'] = 100 * df['e'] / df['y']
    
    predicted_part = df[-prediction_size:]
    
    error_mean = lambda error_name: np.mean(np.abs(predicted_part[error_name]))
    
    return {'MAPE': error_mean('p'), 'MAE': error_mean('e')}
  
PATH = "/content/drive/MyDrive/scpwr/trend/tmp"
folder = os.listdir(PATH)
breakno=0
breaktill=10
N =3 #how many 'hot' keywords are shown
cm_prob_df = pd.DataFrame() #comparing scores set(probability difference between actual value and predicted value)
cm_perdiff_df = pd.DataFrame() #comparing scores set(the percentage difference of actual value to predicted value)
prediction_size = 10 #how many date to predict

for i in folder:
  data = pd.read_csv(os.path.join(PATH,i), sep = ",", parse_dates=["dates"])
  df = data.reset_index()
  df.columns = ['index', 'ds', 'y']
  keyword = i.split(".txt")[0]
  # Split into a train/test set
  train_df = df[:-prediction_size]

  # Initialize and train a model
  m = Prophet()
  m.fit(train_df)

  # Make predictions
  future = m.make_future_dataframe(periods=prediction_size)
  forecast = m.predict(future)
  cmp_df = make_comparison_dataframe(df, forecast)

  '''
  #compute model evaluation
  p_value = sm.tsa.stattools.adfuller(train_df.y)[1]
  if p_value>= 0.5: stationary="non-stationary"
  else: stationary="stationary"
  err = []
  for item in calculate_forecast_errors(cmp_df, prediction_size).items():
    err.append(item)

  #Plot forecast summary
  print('\n\nGraph of frequency prediction for keyword: {0}\n {1}: {2:.3f}\n {3}: {4:.3f}\n p-value={5:.5f}, {6}'.format(
      keyword,err[0][0],err[0][1],err[1][0],err[1][1],p_value,stationary))
  m.plot(forecast)
  plt.show()
  '''

  #-------------------------------------------------
  #compute the comparing score
  '''
  cm_prob: probability difference between actual value and predicted value 
           -> higher difference -> rarer actual value -> "hotter" keyword
  cm_perdiff: the percentage difference of actual value to predicted value
              -> higher difference -> rarer actual value -> "hotter" keyword
  '''
  #-------------------------------------------------


  actual_df = df["y"][-prediction_size:].values #the actual value
  dff = m.setup_dataframe(df, initialize_scales=True)
  sim_values = m.sample_posterior_predictive(dff, vectorized= True)
  predict_ranges = sim_values['yhat'][-prediction_size:] #the range of predition
  predict_mean = cmp_df['yhat'].values[-prediction_size:] #the most possible prediction
  cm_prob = np.zeros(prediction_size) #comparing score(probability difference between actual value and predicted value)
  cm_perdiff = np.zeros(prediction_size) #comparing score(the percentage difference of actual value to predicted value)

  #for every forecast value, find the comparing score(probability of actual value to appear)
  for j in range(prediction_size):
    #the predicted value closetest to the actual value
    yhat_actual = min(predict_ranges[j], key=lambda x:abs(x - actual_df[j]))
    #the prob. of "yhat_actual" in the range of predicted values
    actual_prob = stats.percentileofscore(predict_ranges[j], yhat_actual, kind='rank', nan_policy='propagate')
    #the prob. of predicted value in the predict range
    predict_prob = stats.percentileofscore(predict_ranges[j], predict_mean, kind='rank', nan_policy='propagate')
    #the prob. for comparison
    cm_prob[j] = actual_prob-predict_prob[j]

  #for every forecast value, find the comparing score(the percentage difference of actual value to predicted value)
  for k in range(prediction_size):
    cm_perdiff[k] = (actual_df[k] - predict_mean[k])/predict_mean[k]

  #append to the compare dataframe
  cm_prob_df[keyword] = cm_prob
  cm_perdiff_df[keyword] = cm_perdiff

  breakno+=1
  if breakno>=breaktill: #for test 1 instance only
    break
clear_output()

#
#compare the prob. differences / percentage differences among dates forecasted
#

cm_prob_df = cm_prob_df.set_index(df['ds'][-prediction_size:].values)
print("According to the probability of actual value to appear")
for index,i in cm_prob_df.iterrows():
  print("the {} hottest keywords in {}\n".format(N,index),i.nlargest(n=N))

print("\n\n")

cm_perdiff_df = cm_perdiff_df.set_index(df['ds'][-prediction_size:].values)
print("According to the percentage difference of actual value to predicted value")
for index,i in cm_perdiff_df.iterrows():
  print("the {} hottest keywords in {}\n".format(N,index),i.nlargest(n=N))
