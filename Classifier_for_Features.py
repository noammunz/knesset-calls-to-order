import pandas as pd
import numpy as np
import os
import sys
sys.path.append('/home/nogaschw/Call-to-order')
from tqdm import tqdm
tqdm.pandas()

from itertools import product
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import Dataset, ClassLabel
from sklearn.model_selection import train_test_split
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, precision_recall_fscore_support
import torch.nn.functional as F
import time
from pytorch_lightning.callbacks import TQDMProgressBar

def seed_everything(seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class CustomNet(nn.Module):
    def __init__(self, bert_model_name, num_classes, feature_size=128):
        super(CustomNet, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.dropout = nn.Dropout(0.1)
        bert_hidden_size = self.bert.config.hidden_size

        self.fc_bert = nn.Linear(bert_hidden_size, num_classes)
        self.fc_features_1 = nn.Linear(feature_size, 128)
        self.fc_features_2 = nn.Linear(128, num_classes)

        nn.init.xavier_normal_(self.fc_bert.weight)
        nn.init.constant_(self.fc_bert.bias, 0)
        nn.init.xavier_normal_(self.fc_features_1.weight)
        nn.init.constant_(self.fc_features_1.bias, 0)
        nn.init.xavier_normal_(self.fc_features_2.weight)
        nn.init.constant_(self.fc_features_2.bias, 0)

    def forward(self, input_ids, attention_mask, features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_output.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        logits_bert = self.fc_bert(cls_output)

        features = self.fc_features_1(features)
        features = nn.functional.relu(features)
        features = self.dropout(features)
        logits_features = self.fc_features_2(features)

        return logits_bert, logits_features

class Classifier(pl.LightningModule):
    def __init__(self, model, num_classes, lr=1e-3, alpha=0.5):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.alpha = alpha
        self.test_outputs = []
        self.metrics = {}
        self.val_outputs = []

    def forward(self, input_ids, attention_mask, features):
        return self.model(input_ids, attention_mask, features)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        features = batch['features']
        labels = batch['labels']
        logits_bert, logits_features = self(input_ids, attention_mask, features)
        
        loss_bert = self.loss_fn(logits_bert, labels)
        loss_features = self.loss_fn(logits_features, labels)
        loss = (1 - self.alpha) * loss_bert + self.alpha * loss_features
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        features = batch['features']
        labels = batch['labels']
        logits_bert, logits_features = self(input_ids, attention_mask, features)
        combined_logits = (1 - self.alpha) * logits_bert + self.alpha * logits_features
        loss = self.loss_fn(combined_logits, labels)
        self.val_outputs.append(loss)
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_outputs).mean()
        self.val_outputs = []
        print(f'#####################################################      val_loss: {avg_loss}          #####################################################')


    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        features = batch['features']
        labels = batch['labels']

        logits_bert, logits_features = self(input_ids, attention_mask, features)
        combined_logits = (1 - self.alpha) * logits_bert + self.alpha * logits_features
        self.test_outputs.append({'logits': combined_logits, 'labels': labels})

    def on_test_epoch_end(self):
        logits = torch.cat([x['logits'] for x in self.test_outputs], dim=0)
        labels = torch.cat([x['labels'] for x in self.test_outputs], dim=0).cpu().numpy()
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()
        preds = (probs[:, 1] > 0.5).astype(int)

        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        roc_auc = roc_auc_score(labels, probs[:, 1])
        self.metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': roc_auc}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

class FeatureClassifier:
    def __init__(self, df, feature_col, search, results_path, windows_size, save_model_path=[], epochs=12, batch_size=32,
                 learning_rate=5e-5, max_length=512, alpha=0.5, bert_model_name='onlplab/alephbert-base'):
        self.df = df
        self.feature_col = feature_col
        self.epochs = epochs
        self.search = search
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.alpha = alpha
        self.window_size = windows_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.model = BertModel.from_pretrained(bert_model_name).to(self.device)
        self.num_features = 0
        self.results_path = results_path
        self.save_model_path = save_model_path

        self.preprocess_data()
        self.pipeline()    
    
    def preprocess_data(self):
        self.df = self.df[['conversation', self.feature_col, 'call_to_order']]
        self.df = self.df.rename(columns={'conversation': 'text', self.feature_col: 'features', 'call_to_order': 'labels'})
        self.df = self.df.fillna('')
        ones = self.df[self.df['labels'] == 1]
        zeros = self.df[self.df['labels'] == 0].sample(n=int((ones.shape[0] / 0.2) * 0.8), random_state=42)
        self.df = pd.concat([ones, zeros]).sample(frac=1, random_state=42)
        print(f'number of pos: {len(ones)}, number of neg: {len(zeros)}')
        self.num_features = len(self.df['features'].iloc[0])
    
    def get_dataloaders(self):
        train, test = train_test_split(self.df, test_size=0.2, random_state=42)
        train_dataset = Dataset.from_pandas(train)
        test_dataset = Dataset.from_pandas(test)

        def tokenize(batch):
            return self.tokenizer(batch['text'], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
    
        train_dataset = train_dataset.map(tokenize, batched=True)
        test_dataset = test_dataset.map(tokenize, batched=True)
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'features'])
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'features'])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def pipeline(self):
        model = CustomNet(bert_model_name, num_classes=2, feature_size=self.num_features)
        classifier = Classifier(model, num_classes=2, lr=self.learning_rate, alpha=self.alpha)
        train_loader, test_loader = self.get_dataloaders()
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            logger=False,
            enable_checkpointing=False,
            callbacks=[TQDMProgressBar(refresh_rate=50)]
        )          
        trainer.fit(classifier, train_loader, test_loader)
        trainer.test(classifier, dataloaders=test_loader)

        accuracy, precision, recall, f1, auc = classifier.metrics.values()
        results_df = pd.DataFrame({'search': self.search, 'window_size':self.window_size, 'feature': self.feature_col, 'alpha': alpha, 'epochs': self.epochs, 'accuracy': accuracy,
                                   'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}, index=[0])
        
        if os.path.exists(self.results_path):
            results_df.to_csv(self.results_path, mode='a', header=False, index=False)
        else:
            results_df.to_csv(self.results_path, index=False)

        if self.save_model_path:
            torch.save(classifier.model, self.save_model_path)