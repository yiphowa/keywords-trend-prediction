import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import os
from scipy import stats
from IPython.display import clear_output
%matplotlib inline
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

PATH = "./tmp"
folder = os.listdir(PATH)
breakno=0
breaktill=10
N =3 #how many 'hot' keywords are shown
cm_prob_df = pd.DataFrame() #comparing scores set(probability of the outer region that bearly contains actual value)
cm_perdiff_df = pd.DataFrame() #comparing scores set(the percentage difference of actual value to predicted value)
prediction_size = 10 #how many date to predict

for i in folder:
  data = pd.read_csv(os.path.join(PATH,i), sep = ",", parse_dates=["dates"])
  keyword = i.split(".txt")[0]

  #fit ETS
  datas = data.freq
  datas = datas.set_axis(data.dates)
  model = ETSModel(
    datas,
    error="add",
    trend="add",
    seasonal="add",
    damped_trend=True,
    seasonal_periods=4,
  )
  fit = model.fit()

  #compute evaluation
  p_value = sm.tsa.stattools.adfuller(data.freq)[1]
  if p_value>= 0.5: stationary="non-stationary"
  else: stationary="stationary"
  mape = np.mean(np.abs((datas-fit.fittedvalues)/datas *100))
  
  #predict
  predict_date = 17
  try:
    pred = fit.get_prediction(start="2023-02-{}".format(predict_date-prediction_size+1), end="2023-02-{}".format(predict_date))
  except:
    print(i,"dt have 2023-02-17 data")
    predict_date -= 1
    pred = fit.get_prediction(start="2023-02-{}".format(predict_date-prediction_size+1), end="2023-02-{}".format(predict_date))

  #plot summary
  '''
  df = pred.summary_frame(alpha=0.05) #alpha -> 95% interval
  datas.plot(label="data")
  fit.fittedvalues.plot(label="estimated")
  df["mean"].plot(label="mean prediction")
  df["pi_lower"].plot(linestyle="--", color="tab:blue", label="95% interval")
  df["pi_upper"].plot(linestyle="--", color="tab:blue", label="_")
  plt.ylabel("kw frequency")
  plt.title('Predicted frequency for keyword: {0}\n p-value={1:.5f},  {2}\n MAPE = {3:.3f}%'.format(keyword,p_value,stationary,mape))
  plt.legend()
  plt.show()
  '''

  #-------------------------------------------------
  #compute the comparing score
  '''
  cm_prob: probability of the outer region that barely contains actual value 
          -> lower prob. -> rarer actual value -> "hotter" keyword
  cm_perdiff: the percentage difference of actual value to predicted value
              -> higher difference -> rarer actual value -> "hotter" keyword
  '''
  #-------------------------------------------------

  actual_df = datas[-prediction_size:]#the actual value
  cm_prob = np.zeros(prediction_size) #comparing score(probability of the outer region that bearly contains actual value)
  cm_perdiff = np.zeros(prediction_size) #comparing score(the percentage difference of actual value to predicted value)


  # there is two methods of prob. prediction: 
  # 1. "simulated"(run forecast many times and count the prob.) 
  # 2. "exact"(cal the prob. with variance)
  # most of the prob. prediction uses the "exact" method

  if pred.method == "simulated": 
    #for every forecast value, find the comparing score(probability of the outer region that bearly contains actual value)
    #not sure statistiacally supportive or not
    predict_ranges = pred.simulation_results #the range of predition
    predict_mean = np.mean(pred.simulation_results, axis=1) #the most possible prediction
    for j in range(prediction_size):
      #the predicted value closetest to the actual value
      yhat_actual = min(predict_ranges[j], key=lambda x:abs(x - actual_df[j]))
      #the prob. of "yhat_actual" in the range of predicted values
      actual_prob = stats.percentileofscore(predict_ranges[j], yhat_actual, kind='rank', nan_policy='propagate')
      #the prob. of predicted value in the predict range
      predict_prob = stats.percentileofscore(predict_ranges[j], predict_mean, kind='rank', nan_policy='propagate')
      #the prob. for comparison
      cm_prob[j] = (1-actual_prob)/(1-predict_prob)/2 if actual_prob>predict_prob else -(actual_prob/predict_prob/2)

  else:
    #for every forecast value, find the comparing score(probability of the outer region that bearly contains actual value)
    predict_mean = pred.predicted_mean
    for j in range(prediction_size):
      half_interval_size = np.abs(predict_mean[j] - actual_df[j]) #zscore * var
      q = half_interval_size/np.sqrt(pred.forecast_variance[j]) #zscore
      neg = -1 if (actual_df[j]-predict_mean[j])<0 else 1
      #the prob. for comparison
      cm_prob[j] = (1 - stats.norm.cdf(q))*neg

  #for every forecast value, find the comparing score(the percentage difference of actual value to predicted value)
  cm_perdiff = (actual_df - predict_mean)/predict_mean

  #append to the compare dataframe
  cm_prob_df[keyword] = cm_prob
  cm_perdiff_df[keyword] = cm_perdiff


  breakno+=1
  if breakno>=breaktill: #for test {breaktill} instance only
    break

clear_output()

#
#compare the prob. of outer region / percentage differences among dates forecasted
#

cm_prob_df = cm_prob_df.set_index(data.dates[-prediction_size:].values)
print("According to the probability of the outer region that bearly contains actual value")
for index,i in cm_prob_df.iterrows():
  i2 = i[i.values>=0] #drop out all negative prob. -> no actual value under prediced value
  print("the {} hottest keywords in {}\n".format(N,index),i2.nsmallest(n=N))

print("\n\n")

cm_perdiff_df = cm_perdiff_df.set_index(data.dates[-prediction_size:].values)
print("According to the percentage difference of actual value to predicted value")
for index,i in cm_perdiff_df.iterrows():
  print("the {} hottest keywords in {}\n".format(N,index),i.nlargest(n=N))
