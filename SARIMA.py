import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from scipy.optimize import minimize
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from tqdm import tqdm_notebook
from itertools import product
from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore')
import os

%matplotlib inline

#set up some functions
def optimize_SARIMA(data, parameters_list, d, D, s):
    """
        Return dataframe with parameters and corresponding AIC
        
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order
        D - seasonal integration order
        s - length of season
    """
    
    results = []
    best_aic = float('inf')
    
    for param in tqdm_notebook(parameters_list):
        try: model = sm.tsa.statespace.SARIMAX(data, order=(param[0], d, param[1]),
                                               seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
            
        aic = model.aic
        
        #Save best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])
        
    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    #Sort in ascending order, lower AIC is better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    
    return result_table

def find_mape(y_true,y_pred):
  y_true,y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs(y_true - y_pred)/y_true)*100

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#Set initial values and some bounds
ps = [0, 1, 2, 4, 6]
d = 1
qs = range(1, 3)
Ps = [0, 1, 2, 4, 6]
D = 1
Qs = range(1, 3)
s = 5

#total iteration = 5*2*5*2 = 100

#Create a list with all possible combinations of parameters
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)


PATH = "./tmp"
folder = os.listdir(PATH)
breakno=0
breaktill=10
N =3 #how many 'hot' keywords are shown
cm_prob_df = pd.DataFrame() #comparing scores set(probability difference between actual value and predicted value)
cm_perdiff_df = pd.DataFrame() #comparing scores set(the percentage difference of actual value to predicted value)
prediction_size = 10 #how many date to predict

for i in folder:
  data = pd.read_csv(os.path.join(PATH,i), sep = ",", parse_dates=["dates"])

  #test p value
  p_value = sm.tsa.stattools.adfuller(data.freq)[1]
  if p_value>= 0.5: stationary="non-stationary"
  else: stationary="stationary"

  result_table = optimize_SARIMA(data.freq, parameters_list, d, D, s) #around 3mins to compute

  #set the best model
  p, q, P, Q = result_table.parameters[0]
  aic = result_table.aic[0]
  best_model = sm.tsa.statespace.SARIMAX(data.freq, order=(p, d, q),
                                       seasonal_order=(P, D, Q, s)).fit(disp=-1)

  #predict
  try:
    pred = best_model.get_prediction(start=-prediction_size, end=-1)
  except:
    print(i,"dt have 2023-02-17 data")
    pred = best_model.get_prediction(start=-prediction_size-1, end=-2)
 
  '''
  #for stat info
  print(best_model.summary())
  # Make a dataframe containing actual and predicted prices
  comparison = pd.DataFrame({'actual': data.freq.values[-prediction_size:],
                  'prediction':pred.predicted_mean.values},index = pd.date_range(start='2023-02-08', periods=10,))

  #find the MAPE of the model
  mape = find_mape(comparison.actual,comparison.prediction)
  aic = result_table.aic[0]
  if p_value>= 0.5: stationary="non-stationary"
  else: stationary="stationary"

  #Plot prediction vs actual price
  plt.figure(figsize=(17, 8))
  plt.plot(comparison.actual, label="actual")
  plt.plot(comparison.prediction, label="prediction")
  plt.title('Predicted frequency for keyword: {0}\n p-value={1:.5f},  {2} \nMAPE={3:.3f}%,  AIC={4:.3f}'.format(i, p_value, stationary, mape, aic))
  plt.ylabel('kw frequency')
  plt.xlabel('date')
  plt.legend(loc='best')
  plt.grid(False)
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
  actual_df = data.freq.values[-prediction_size:]#the actual value
  cm_prob = np.zeros(prediction_size) #comparing score(probability of the outer region that bearly contains actual value)
  cm_perdiff = np.zeros(prediction_size) #comparing score(the percentage difference of actual value to predicted value)

  #for every forecast value, find the comparing score(probability of the outer region that bearly contains actual value)
  predict_mean = pred.predicted_mean.values
  for j in range(prediction_size):
    half_interval_size = np.abs(predict_mean[j] - actual_df[j]) #zscore * var  #(var = pred.se_mean())
    q = half_interval_size/pred.se_mean.values[j] #zscore
    neg = -1 if (actual_df[j]-predict_mean[j])<0 else 1
    #the prob. for comparison
    cm_prob[j] = (1 - pred.dist.cdf(q))*neg

  #for every forecast value, find the comparing score(the percentage difference of actual value to predicted value)
  cm_perdiff = (actual_df - predict_mean)/predict_mean

  #append to the compare dataframe
  cm_prob_df[keyword] = cm_prob
  cm_perdiff_df[keyword] = cm_perdiff


  breakno+=1
  if breakno>=breaktill: #for test instance only
    break
clear_output()
  
#
#compare the prob. differences / percentage differences among dates forecasted
#

cm_prob_df = cm_prob_df.set_index(data.dates[-prediction_size:].values)
print("According to the probability of actual value to appear")
for index,i in cm_prob_df.iterrows():
  i2 = i[i.values>=0] #drop out all negative prob. -> no actual value under prediced value
  print("the {} hottest keywords in {}\n".format(N,index),i2.nsmallest(n=N))

print("\n\n")

cm_perdiff_df = cm_perdiff_df.set_index(data.dates[-prediction_size:].values)
print("According to the percentage difference of actual value to predicted value")
for index,i in cm_perdiff_df.iterrows():
  print("the {} hottest keywords in {}\n".format(N,index),i.nlargest(n=N))
