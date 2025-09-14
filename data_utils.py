import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

#dataset reading
def read_data(path, labels = 1):
  data_frame = pd.read_csv(path)
  #print(labels)
  if labels == 1:
    columns = data_frame.columns.tolist()[:2]
    data_frame = data_frame[columns]

  return data_frame

def read_amrita(path , type='pt', labels = 1):
  train = read_data(f'{path}/train.csv', labels)
  val = read_data(f'{path}/val.csv', labels)
  test = read_data(f'{path}/test.csv', labels)

  if type == 'tf':
    data = pd.concat([
      train,
      test,
      val,
    ]).reset_index(drop = True)
    
    return data
  else:
    return train, val, test



#handle data imbalance
def handle_imbalance(data_frame, labels = 1):
  data_details = {} #return details of data
  if labels == 1:
    columns = data_frame.columns.tolist()[:2]
    DGA_df = data_frame[data_frame[columns[1]] == 0]

    benign_df = data_frame[data_frame[columns[1]] > 0]
    non_DGA_samples = benign_df.shape[0]
    df = pd.concat([
      DGA_df.sample(n= benign_df, random_state=1),
      benign_df
    ])
    for col in df.columns.to_list()[1:]:
      data_details[col] = df[col].value_counts()[1]
  
  return df, data_details

#draw data categories bar plotting
def draw_categories(data_frame, labels = 1): 
  if labels == 1:
    columns = data_frame.columns.tolist()[:2]
    DGA_df = data_frame[data_frame[columns[1]] == 0]
    benign_df = data_frame[data_frame[columns[1]] > 0]
    pd.DataFrame(dict(
      DGA=[len(DGA_df)], 
      benign=[len(benign_df)]
    )).plot(kind='barh');
  else:
    columns = data_frame.columns.tolist()[0:-1]
    benign_df = data_frame[data_frame[columns[1]] == 0]
    DGA_df = data_frame[data_frame[columns[1]] > 0]
    pd.DataFrame(dict(
      DGA=[len(DGA_df)], 
      benign=[len(benign_df)]
    )).plot(kind='barh');

#test_tensorflow_models

