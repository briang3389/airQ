import pandas as pd
import numpy as np

originaldf = pd.read_excel('data/train_data.xlsx')

class DataPrep:
    # must contain each one of these labels
    # collumns of interest
    input_data_cols = ["CO(GT)", "PT08.S1(CO)", "NMHC(GT)"] # add features to end to make itself predict the output col - lstm not limit to 1 feature
    # input_data_cols = ["sales_amount"]
    output_data_cols = ["CO(GT)"]
    percent = 0.8 # test-train split percentage
    lookback = 20 # number of units used to make prediction
    predict = 5 # number of units that will be predicted

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

train_data = np.asarray([dataSetInput, dataSetOutput], dtype=object)
test_data = np.asarray([dataSetInput, dataSetOutput], dtype=object)

np.save('./data/processed_train_data', train_data)
np.save('./data/processed_test_data', test_data)

with open("data/norm_params.json", "w") as f:
    f.write("normalized")
