import torch
import tensorflow as tf
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import time
import datetime
import timeit


#utils functions 
def get_tf_device():
  #return either '/device:GPU:0' or '/cpu:0'
  device_name = tf.test.gpu_device_name()
  return device_name

def get_pt_device():
  #return either 0 for cpu or number of gpus
  AVAIL_GPUS = min(1, torch.cuda.device_count())
  return AVAIL_GPUS

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = (round((elapsed), 3))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def result_df(training_stats):
  # Create a DataFrame from our training statistics.
  df_stats = pd.DataFrame(data=training_stats)


  # Use the 'epoch' as the row index.""
  df_stats = df_stats.set_index('Model_Name')

  # Display floats with three decimal places.
  pd.set_option('precision',3)

  # A hack to force the column headers to wrap.
  #df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

  
  return df_stats



