import pandas as pd
import numpy as np
from constants import *

originaldf = pd.read_excel('data/data.xlsx')
originaldf.replace({-200: np.NaN}, inplace=True)
originaldf.replace({'-200': np.NaN}, inplace=True)
#originaldf.interpolate(method='linear', inplace=True)
for i in range(2, len(list(originaldf.columns))):
    originaldf[list(originaldf.columns)[i]] = originaldf[list(originaldf.columns)[i]].interpolate()

class DataPrep:
    # must contain each one of these labels
    # collumns of interest
    input_data_cols = INPUT_DATA_COLS # add features to end to make itself predict the output col - lstm not limit to 1 feature
    # input_data_cols = ["sales_amount"]
    output_data_cols = OUTPUT_DATA_COLS
    percent = PERCENT_TEST_TRAIN # test-train split percentage
    lookback = LOOKBACK # number of units used to make prediction
    predict = PREDICT # number of units that will be predicted
    shuffle_train_data = SHUFFLE_TRAIN_TEST_DATA # whether you want to shuffle data during training or not

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df = originaldf.copy()
df[DataPrep.input_data_cols] = scaler.fit_transform(originaldf[DataPrep.input_data_cols])

'''
import matplotlib.pyplot as plt
plt.plot(originaldf["CO(GT)"])
plt.show()

plt.plot(df["CO(GT)"])
plt.show()

testdf = df.copy()
testdf[DataPrep.input_data_cols] = scaler.inverse_transform(df[DataPrep.input_data_cols])
plt.plot(testdf["CO(GT)"])
plt.show()
'''

print("number of rows: " + str(len(df.index)))

inputcolsdf = df[DataPrep.input_data_cols].copy()
outputcolsdf = df[DataPrep.output_data_cols].copy()
dataSetInput = []
dataSetOutput = []
for i in range(len(df.index)-(DataPrep.lookback+DataPrep.predict-1)):
  #print(i, " ", i+DataPrep.lookback, " ", i+DataPrep.lookback+DataPrep.predict)
  dataSetInput.append(inputcolsdf.iloc[list(range(i, i+DataPrep.lookback)),:].to_numpy())
  dataSetOutput.append(outputcolsdf.iloc[list(range(i+DataPrep.lookback, i+DataPrep.lookback+DataPrep.predict)),:].to_numpy())

'''
train_data = np.asarray([dataSetInput, dataSetOutput], dtype=object)
test_data = np.asarray([dataSetInput, dataSetOutput], dtype=object)

np.save('./data/processed_train_data', train_data)
np.save('./data/processed_test_data', test_data)
'''

with open("data/norm_params.json", "w") as f:
    f.write("normalized")
    
    
def split_test_train_data(ts_inp,ts_out,shuffle = False,percent_train = DataPrep.percent):
  len_data = len(df.index)
  print("shuffle data:",shuffle)
  print(ts_out[0])
  if shuffle:
    all_data_arr = np.array((ts_inp,ts_out),dtype=object).T
    #print(all_data_arr.shape)
    np.random.seed(RANDOM_SPLIT_SEED)
    np.random.shuffle(all_data_arr)
    all_data_arr = all_data_arr.T
    #print(all_data_arr.shape)
    ts_inp,ts_out = all_data_arr[0],all_data_arr[1]
  print(ts_out[0])
    #for x in rand_arr_indexes[0:int(percent*len_data)]:
    #  train_x.append()
      

  #y is lookback, x is prediction
  train_x = np.asarray(ts_inp[0:int(DataPrep.percent*len_data)])
  print("train_x",train_x.shape)
  train_y = np.asarray(ts_out[0:int(DataPrep.percent*len_data)])
  print("train_y",train_y.shape)
  valid_x = np.asarray(ts_inp[int(DataPrep.percent*len_data):])
  print("valid_x",valid_x.shape)
  valid_y = np.asarray(ts_out[int(DataPrep.percent*len_data):])
  print("valid_y",valid_y.shape)

  return train_x,train_y,valid_x,valid_y
  
train_x,train_y,valid_x,valid_y = split_test_train_data(dataSetInput.copy(),dataSetOutput.copy(),shuffle = DataPrep.shuffle_train_data)

np.save('./data/processed_train_prediction', train_x)
np.save('./data/processed_train_lookback', train_y)
np.save('./data/processed_validation_prediction', valid_x)
np.save('./data/processed_validation_lookback', valid_y)

