import torch
import torch.nn as nn
import subprocess


class pt_initialize:
    '''
        This class is used for project initialization
    '''
    TASK_NAME= 'DGA'
    MAX_LENGTH = 75
    MAX_EPOCH = 5
    TRAIN_BATCH_SIZE = 64
    EVAL_BATCH_SIZE = 64
    MODEL_NAME = "distilroberta-base" #"distilbert-base-uncased" #"bert-base-uncased" 
    LABELS_NO = 21
    LEARNING_RATE = 5e-5
    ADAM_EPSILON = 1e-8
    RANDOM_SEED = 42
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    def __init__(self,
                 LABELS_NO: int =2,
                 MAX_LENGTH: int = 75,
                 MAX_EPOCH: int = 5,
                 TRAIN_BATCH_SIZE: int = 64,
                 EVAL_BATCH_SIZE: int = 64,
                 LEARNING_RATE: float = 5e-5,
                 ADAM_EPSILON: float = 1e-8,
                 MODEL_NAME: str = "distilbert-base-uncased",
                 **kwargs,
                ):
        #self.TASK_NAME= TASK_NAME
        self.MAX_LENGTH = MAX_LENGTH
        self.MAX_EPOCH = MAX_EPOCH
        self.TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE
        self.EVAL_BATCH_SIZE = EVAL_BATCH_SIZE
        self.MODEL_NAME = MODEL_NAME
        self.LABELS_NO = LABELS_NO
        self.LEARNING_RATE = LEARNING_RATE
        self.ADAM_EPSILON = ADAM_EPSILON
        
        
    def show_gpu_details(self, msg: str='Initial GPU memory usage:'):
        """
        ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
        """
        if torch.cuda.is_available():    
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
            def query(field):
                return(subprocess.check_output(
                    ['nvidia-smi', f'--query-gpu={field}',
                        '--format=csv,nounits,noheader'], 
                    encoding='utf-8'))
            def to_int(result):
                return int(result.strip().split('\n')[0])

            used = to_int(query('memory.used'))
            total = to_int(query('memory.total'))
            pct = used/total
            print('\n' + msg, f'{100*pct:2.1f}% ({used} out of {total})') 
        else:
            print('No GPU available, using the CPU instead.')

class tf_initialize:
  
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

