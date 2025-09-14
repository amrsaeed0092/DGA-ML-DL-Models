import pandas as pd
import numpy as np
import pytorch_lightning as pl
try:
    from torchmetrics.functional import accuracy, f1_score as f1, auroc
except ImportError:
    from pytorch_lightning.metrics.functional import accuracy, f1, auroc
#from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaTokenizerFast , BertConfig, BertModel,RobertaModel, AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel, AutoConfig

#import custom modules
import sys
sys.path.insert(0,'/content/drive/MyDrive/Colab Notebooks/DGA_Version_70length/')

from init import pt_initialize as Initialize

RANDOM_SEED = 42
pl.seed_everything(RANDOM_SEED)

#define dataset modules
class DGADataset(Dataset):
    def __init__(
        self, 
        data: pd.DataFrame, 
        tokenizer: AutoTokenizer, 
        max_token_len: int = 128,
        categories = None
      ):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
        self.categories = categories

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        domain_name = data_row.domain
        labels = data_row[self.categories]

        encoding = self.tokenizer.encode_plus(
          domain_name,
          add_special_tokens=True,
          max_length=self.max_token_len,
          return_token_type_ids=False,
          padding="max_length",
          truncation=True,
          return_attention_mask=True,
          return_tensors='pt',
        )

        return dict(
          domain_name=domain_name,
          input_ids=encoding["input_ids"].flatten(),
          attention_mask=encoding["attention_mask"].flatten(),
          labels=torch.FloatTensor(labels)
        )

class DGADataModule(pl.LightningDataModule):

    def __init__(self, train_df,val_df, test_df, tokenizer, train_batch_size=32,val_batch_size=32,test_batch_size=32, max_token_len=128, LABELS=None):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.LABELS = LABELS

        
    def setup(self, stage=None):
        self.train_dataset = DGADataset(self.train_df,
                                          self.tokenizer,
                                          self.max_token_len,
                                          self.LABELS
                                         )
        self.val_dataset = DGADataset(self.val_df,
                                       self.tokenizer,
                                       self.max_token_len,
                                       self.LABELS
                                      )
        self.test_dataset = DGADataset(self.test_df,
                                       self.tokenizer,
                                       self.max_token_len,
                                       self.LABELS
                                      )
        
    def train_dataloader(self):
        return DataLoader(
          self.train_dataset,
          batch_size=self.train_batch_size,
          shuffle=True,
          num_workers=2
        )

    
    def val_dataloader(self):
        return DataLoader(
          self.val_dataset,
          batch_size=self.val_batch_size,
          num_workers=2
        )

    
    def test_dataloader(self):
        return DataLoader(
          self.test_dataset,
          batch_size=self.test_batch_size,
          num_workers=2
        )

#define the dga pytorch lightning model
class DGATransformer(pl.LightningModule):
    def __init__(self, LABEL_COLUMNS = None, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.LABEL_COLUMNS = LABEL_COLUMNS
        n_classes = len(LABEL_COLUMNS)
        #model_name = 'haisongzhang/roberta-tiny-cased'
        self.bert = AutoModel.from_pretrained(Initialize.MODEL_NAME, 
                                              return_dict=True,  
                                              #from_tf=True,
                                              #n_layers = 2,
                                              num_hidden_layers = 4,
                                              #hidden_size = 256,
                                              #num_attention_heads = 4

                                             )
        
        self.conf = AutoConfig.from_pretrained(Initialize.MODEL_NAME) #Initialize.MODEL_NAME
        embed_dim = self.bert.config.hidden_size / 2
        classifier_dropout = self.conf.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)

        #self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
       
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.bert.config.hidden_size),
            nn.Linear(self.bert.config.hidden_size, n_classes)
        )
        
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion =  nn.BCELoss()

        # Initialize weights and apply final processing
        #self.post_init()


    def forward(self, input_ids, attention_mask, labels=None):
        #print(input_ids.shape, attention_mask.shape, labels.shape )
        output = self.bert(input_ids, attention_mask=attention_mask)
        pooler_output = output[1]
        pooler_output = self.dropout(pooler_output)
        #output = self.classifier(pooler_output)
        #print('Amr')
        output = self.mlp_head(pooler_output)
        #print('Amr')
        output = torch.sigmoid(output)    
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):

        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        for i, name in enumerate(self.LABEL_COLUMNS):
            class_roc_auc = auroc(predictions[:, i], labels[:, i])
            self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)


    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=2e-5)

        scheduler = get_linear_schedule_with_warmup(
          optimizer,
          num_warmup_steps=self.n_warmup_steps,
          num_training_steps=self.n_training_steps
        )

        return dict(
          optimizer=optimizer,
          lr_scheduler=dict(
            scheduler=scheduler,
            interval='step'
          )
        )


      
