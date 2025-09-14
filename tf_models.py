
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Input, concatenate, GlobalMaxPool1D, Conv1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score
import numpy as np
import pandas as pd
import re

from sklearn.metrics import roc_curve, auc
from itertools import cycle

#initialization
MAX_STRING_LENGTH = 256 #after padding
MAX_INDEX = 70  #dictionary max character
max_epoch = 5
batch_size = 64
EMBEDDING_DIMENSION = 128
NUM_CONV_FILTERS = 60
max_features = 38
input_shape = MAX_STRING_LENGTH
net = {}
n_classes = 1

#create models
def build_highnam(n_classes = 1):
  #Highnam, K., Puzio, D., Luo, S., Jennings, N.R.: Real-time detection of dictionary
  #dga network traffic using deep learning. SN Computer Science 2(2), 1–17 (2021)

  net['input'] = Input((input_shape,), dtype='int32', 
                      name='input')

  ########################
  #          CNN         #
  ########################

  net["embeddingCNN"] = Embedding(
                          output_dim=EMBEDDING_DIMENSION, 
                          input_dim=MAX_INDEX,
                          input_length=MAX_STRING_LENGTH, 
                          name='embeddingCNN')(net["input"])

  # Parallel Convolutional Layer

  net["conv2"] = Conv1D(NUM_CONV_FILTERS, 2, name="conv2")(net["embeddingCNN"])

  net["conv3"] = Conv1D(NUM_CONV_FILTERS, 3, name="conv3")(net["embeddingCNN"])

  net["conv4"] = Conv1D(NUM_CONV_FILTERS, 4, name="conv4") (net["embeddingCNN"])

  net["conv5"] = Conv1D(NUM_CONV_FILTERS, 5, name="conv5") (net["embeddingCNN"])

  net["conv6"] = Conv1D(NUM_CONV_FILTERS, 6, name="conv6")  (net["embeddingCNN"])

  # Global max pooling

  net["pool2"] = GlobalMaxPool1D(name="pool2") (net["conv2"])

  net["pool3"] = GlobalMaxPool1D(name="pool3")   (net["conv3"])

  net["pool4"] = GlobalMaxPool1D(name="pool4")   (net["conv4"])

  net["pool5"] = GlobalMaxPool1D(name="pool5")     (net["conv5"])

  net["pool6"] = GlobalMaxPool1D(name="pool6") (net["conv6"])


  net["concatcnn"] = concatenate([net["pool2"],
                              net["pool3"], net["pool4"],
                              net["pool5"], net["pool6"]], 
                              axis=1, name='concatcnn')


  net["dropoutcnnmid"] = Dropout(0.5, name="dropoutcnnmid")  (net["concatcnn"])

  net["densecnn"] = Dense(NUM_CONV_FILTERS, activation="relu",
                          name="densecnn")(net["dropoutcnnmid"])

  net["dropoutcnn"] = Dropout(0.5, name="dropoutcnn")   (net["densecnn"])

  ########################
  #         LSTM         #
  ########################

  net["embeddingLSTM"] = Embedding(output_dim=max_features, 
                              input_dim=256,
                              input_length=MAX_STRING_LENGTH, 
                              name='embeddingLSTM') (net["input"])


  net["lstm"] = LSTM(256, name="lstm")(net["embeddingLSTM"])

  net["dropoutlstm"] = Dropout(0.5, name="dropoutlstm")  (net["lstm"])

  ########################
  #    Combine - ANN     #
  ########################

  net['concat'] = concatenate([net['dropoutcnn'], 
                              net['dropoutlstm']], 
                              axis=-1, name='concat')

  net['dropoutsemifinal'] = Dropout(0.5, name="dropoutsemifinal") (net['concat'])

  net['extradense'] = Dense(100, activation='relu',  name="extradense") (net['dropoutsemifinal'])

  net['dropoutfinal'] = Dropout(0.5, name="dropoutfinal") (net['extradense'])

  net['output'] = Dense(n_classes, activation='sigmoid', name="output")  (net['dropoutfinal'])

  model = Model(net['input'], net['output'])
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

  return model


def build_LSTM_MI_model(MAX_INDEX, MAX_STRING_LENGTH, n_classes = 1):
  """Build LSTM model for two-class classification"""
  model = Sequential()
  model.add(Embedding(MAX_INDEX, 128, input_length=MAX_STRING_LENGTH))
  model.add(LSTM(128))
  model.add(Dropout(0.5))
  model.add(Dense(n_classes))
  model.add(Activation('sigmoid'))

  model.compile(loss='binary_crossentropy',optimizer='rmsprop')

  return model


def build_CBDC(n_classes=1):
  net['input'] = Input((input_shape,), 
                    name='input')

  net["embeddingCNN"] = Embedding(
                              output_dim=EMBEDDING_DIMENSION, 
                              input_dim=MAX_INDEX,
                              input_length=MAX_STRING_LENGTH, 
                              name='embeddingCNN') (net["input"])
  # Parallel Conv Filters

  net["conv2"] = Conv1D(NUM_CONV_FILTERS, 2, name="conv2") (net["embeddingCNN"])

  net["conv3"] = Conv1D(NUM_CONV_FILTERS, 3, name="conv3") (net["embeddingCNN"])

  net["conv4"] = Conv1D(NUM_CONV_FILTERS, 4, name="conv4") (net["embeddingCNN"])

  net["conv5"] = Conv1D(NUM_CONV_FILTERS, 5, name="conv5") (net["embeddingCNN"])

  net["conv6"] = Conv1D(NUM_CONV_FILTERS, 6, name="conv6") (net["embeddingCNN"])

  # Global max pooling operation for each filter size

  net["pool2"] = GlobalMaxPool1D(name="pool2") (net["conv2"])

  net["pool3"] = GlobalMaxPool1D(name="pool3") (net["conv3"])

  net["pool4"] = GlobalMaxPool1D(name="pool4") (net["conv4"])

  net["pool5"] = GlobalMaxPool1D(name="pool5") (net["conv5"])

  net["pool6"] = GlobalMaxPool1D(name="pool6") (net["conv6"])


  net["concatcnn"] = concatenate([net["pool2"],
                          net["pool3"], net["pool4"],
                          net["pool5"], net["pool6"]], 
                          axis=1, name='concatcnn')

  net["dropoutcnnmid"] = Dropout(0.5, 
                      name="dropoutcnnmid")(net["concatcnn"])

  net["densecnn"] = Dense(NUM_CONV_FILTERS, activation="relu", 
                      name="densecnn")(net["dropoutcnnmid"])

  net["dropoutcnn"] = Dropout(0.5, 
                      name="dropoutcnn")(net["densecnn"])

  net["output"] = Dense(n_classes, activation="sigmoid", 
                      name="densefinal")(net["dropoutcnn"])


  model = Model(net['input'], net['output'])
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

  return model


def prepare_tokenizer(maxlen = 256):
	'''
		return tokenizer object and max_index
	'''
	alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
	char_dict = {}
	for i, char in enumerate(alphabet):
		char_dict[char] = i + 1

		
	#Maximum number of dictionary samples
	max_index = max(char_dict.values())+1
	
	# Tokenizer Initialization
	tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
	# Use char_dict to replace the tk.word_index
	tk.word_index = char_dict 
	# Add 'UNK' to the vocabulary 
	tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
	
	print('Long of characters dictionary', max(char_dict.values()))
	
	return tk, max_index
	

def text_2_sequence(data_frame, tk, maxlen = 256, LABELS_NO = 1):
  '''
    return X, y, class_list, classes
  '''
  texts = data_frame['domain'].values # texts contrain all setences
  texts = [s.lower() for s in texts] # convert to lower case 

  #convert text to numerical sequences
  sequences = tk.texts_to_sequences(texts)

  ## Padding the sequences to 128 maxlength
  data = pad_sequences(sequences, maxlen=maxlen , padding='post')

  ## convert data into numpy array
  data = np.array(data)

  #compute output classes
  if LABELS_NO == 1:
    class_list = data_frame['benign'].values
    classes = to_categorical(class_list)
    y=np.array(data_frame['benign'])
  else:
    c = data_frame.columns.tolist()[1:]
    class_list = data_frame.columns.tolist()[1:]
    classes = len(class_list)
    y=np.array(data_frame[class_list])
    

  #convert data into arrays for processing
  X = np.array(data)

  return X, y, class_list, classes
	


