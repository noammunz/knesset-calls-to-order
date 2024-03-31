import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import os
import gc
import sys
sys.path.append('/home/nogaschw/Call-to-order')
from Preprocessing import Preprocessing 
from WindowsData import WindowsData
from itertools import product

class Pipeline:
    def __init__(self, model_name, epochs=7, batch_size=32, learning_rate=5e-5, max_length=512):
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(self.device)
        
    @staticmethod
    def seed_everything(seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def seconds_to_mm_ss(seconds):
        minutes = '0' + str(int(seconds // 60)) if seconds // 60 < 10 else str(int(seconds // 60))
        seconds = str(round(seconds % 60, 2)) if seconds % 60 >= 10 else '0' + str(round(seconds % 60, 2))
        return minutes + ":" + seconds
    
    def preprocess_data(df, resample=False):
        df = df[['conversation', 'call_to_order']]
        df.fillna('', inplace=True)
        df = df.rename(columns={'conversation': 'text', 'call_to_order': 'label'})
        
        if resample:
            ones = df[df['label'] == 1]
            print(len(ones))
            zeros = df[df['label'] == 0].sample(n=int((ones.shape[0] / 0.2) * 0.8), random_state=42)
            df = pd.concat([ones, zeros]).sample(frac=1, random_state=42)

        df.to_csv('/home/nogaschw/Call-to-order/Data/undersampled.csv', index=False)
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        return train, test

    def tokenize(self, batch):
        return self.tokenizer(batch['text'], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
    
    def get_dataloaders(self, train, test):
        train_dataset = Dataset.from_pandas(train)
        test_dataset = Dataset.from_pandas(test)
        train_dataset = train_dataset.map(self.tokenize, batched=True)
        test_dataset = test_dataset.map(self.tokenize, batched=True)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def train_model(self, train_loader):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        weights = torch.tensor([0.1, 0.9], dtype=torch.float32).to(self.device)
        criterion = nn.CrossEntropyLoss()#weight=weights)
        for epoch in range(self.epochs):
            self.model.train()
            i = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids =torch.stack(batch['input_ids'], dim=1).to(self.device)
                attention_mask = torch.stack(batch['attention_mask'], dim=1).to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print(f'{i} Epoch: {epoch}, Loss: {loss.item()}')
                i += 1
        return self.model

    def evaluate_model(self, test_loader, log_path='/home/nogaschw/Call-to-order/Data/samples.csv', log=False):
        self.model.eval()
        y_true, y_pred, y_probs = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids =torch.stack(batch['input_ids'], dim=1).to(self.device)
                attention_mask = torch.stack(batch['attention_mask'], dim=1).to(self.device)
                labels = batch['label'].to(self.device)
                output = self.model(input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(output.logits, 1)
                probs = torch.nn.functional.softmax(output.logits, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_probs.extend(probs.cpu().numpy())

                if log:
                    texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                    log_df = pd.DataFrame({'text': texts, 'label': labels.cpu().numpy(), 'predicted': predicted.cpu().numpy()})
                    if os.path.exists(log_path):
                        log_df.to_csv(log_path, mode='a', header=False, index=False)
                    else:
                        log_df.to_csv(log_path, index=False)

        return y_true, y_pred, y_probs
    
    def _verbose(self, text, verbose=True):
        if verbose:
            print(text)

    def run(self, train, test, verbose=True, do_train=True, log_samples=False):
        start_time = time()
        self._verbose('Getting dataloaders...', verbose)
        train_loader, test_loader = self.get_dataloaders(train, test)
        self._verbose('Training model...', verbose)
        if do_train:
            self.train_model(train_loader)
        else:
            self.model.load_state_dict(torch.load('/home/nogaschw/Call-to-order/Data/4_window_undersampling.pth'))
        self._verbose('Evaluating model...', verbose)
        y_true, y_pred, y_probs = self.evaluate_model(test_loader, log=log_samples)
        time_elapsed = self.seconds_to_mm_ss(time() - start_time)
        return self.model, y_true, y_pred, y_probs, time_elapsed
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

        
    def get_metrics(self, y_true, y_pred, y_prob, window_size, time, search='-', df_path='/home/nogaschw/Call-to-order/Data/metrics23.csv'):
        if not os.path.exists(df_path):
            columns = ['search', 'F1 Score', 'Precision', 'Recall', 'ROC AUC', 'Accuracy', 'window size', 'time', 'epochs', 'batch size', 'learning rate']
            df = pd.DataFrame(columns=columns)
        else:
            df = pd.read_csv(df_path)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, np.array(y_prob)[:, 1])

        new_row = {'search': search, 'F1 Score': f1, 'Precision': precision, 'Recall': recall, 'ROC AUC': roc_auc, 'Accuracy': accuracy, 'window size': window_size, 'time': time,
                    'epochs': self.epochs, 'batch size': self.batch_size, 'learning rate': self.learning_rate}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df.to_csv(df_path, index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    print(f'Starting...')
    Pipeline.seed_everything(42)

    # parameters
    model_name = "onlplab/alephbert-base"
    window_size = int(sys.argv[1])
    save_name = f'{window_size}_window_undersampling.pth'
    log_samples = True
    do_train = True
    
    print(f'Loading data...')
    data = Preprocessing(data_path="Data/preprocessed_data.csv", save_path="Data/", base_path="/home/nogaschw/Call-to-order")
    df = WindowsData(data.data, save_path=data.save_path, windows_size=window_size).data
    train, test = Pipeline.preprocess_data(df, resample=True)
    
    print(f'Running pipeline...')
    pipeline = Pipeline(model_name)
    model, y_true, y_pred, y_probs, time_elapsed = pipeline.run(train, test, log_samples=log_samples, do_train=do_train)
    # save model
    pipeline.save_model('/home/nogaschw/Call-to-order/Data/' + save_name)

    # get metrics
    pipeline.get_metrics(y_true, y_pred, y_probs, window_size, time_elapsed, search=0)

