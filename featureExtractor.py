import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import sys
sys.path.append('/home/nogaschw/Call-to-order')
from Preprocessing import Preprocessing 
from WindowsData import WindowsData
from transformers import pipeline
from tokenizers.decoders import WordPiece
from tqdm import tqdm
tqdm.pandas()
import pickle as pkl
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
# import torch lightning
import pytorch_lightning as pl
from sklearn.preprocessing import OneHotEncoder

class FeatureExtractor:
    """
    FeatureExtractor class:
        - data_path: str, path to the data file
        - window_size: int, size of the window to use for the windowed data
        - df: pd.DataFrame, the data
        - add_features: method, adds all the features to the data
        - save_df: method, saves the data to a csv file
        - get_windowed_df: method, gets the windowed data
        - add_chairperson_features: method, adds the chairperson features to the data
        - add_speaker_features: method, adds the speaker features to the data
        - add_committee_features: method, adds the committee features to the data
        - save_path: str, path to save the data to
    """
    def __init__(self, data_path, window_size=9):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.window_size = window_size
        self.add_features()
        self.save_df(save_path)
    
    def save_df(self, save_path):
        self.df.to_csv(save_path, encoding='utf-8-sig', index=False)

    def add_features(self, verbose=True):
        if verbose:
            print('getting windowed df...')
        self.get_windowed_df()
        if verbose:
            print('adding chairperson features...')
        self.add_chairperson_features()
        if verbose:
            print('adding speaker features...')
        self.add_speaker_features()
        if verbose:
            print('adding committee features...')
        self.add_committee_features()
        if verbose:
            print('done adding features.')

    def get_windowed_df(self):
        windowed_df = WindowsData(self.df, 'temp.csv', self.window_size).data
        try:
            windowed_df = windowed_df.drop(columns=['Unnamed: 0'])
        except:
            pass
        self.df = windowed_df

    def add_chairperson_features(self):
        self.df.loc[self.df['session_id'] == 49646, 'chairperson'] = 'ווליד טאהא'

        chairperson_map = {
            '': '', '': '',
            '': '', '': '',
            'אופיר פז-פינס': 'אופיר פינס', 'אופיר פינס-פז': 'אופיר פינס',
            'אי יי שפירא': "אי י' שפירא", "ג' גל": 'גי גל',
            "ח' קורפו": 'חי קורפו', 'י י מצא': 'י. מצא',
            "י' מצא": 'י. מצא', 'כרמל שאמה': 'כרמל שאמה-הכהן',
            'כרמל שאמה הכהן': 'כרמל שאמה-הכהן', "מ. רייסר": 'מי רייסר',
            "מ' גפני": 'משה גפני', 'מ"ז פלדמן': 'מ.ז. פלדמן',
            'מל פולישוק-בלוך': 'מלי פולישוק בלוך', 'מלי פולישוק-בלוך': 'מלי פולישוק בלוך',
            "מנחם בן-ששון": "מנחם בן ששון", "ס' טריף": "סאלח טריף", "עמנואל זיסמו":"עמנואל זיסמן", 
            "רוחמה אברהם בלילא":"רוחמה אברהם", "רוני  בר-און":"רוני בר-און", "רוני בראון":"רוני בר-און",
            "ש' שמחון":"שלום שמחון", "ד' ליבאי":"ד. ליבאי", "י' ליצמן":"יעקב ליצמן", "קולט אביטל":"אביטל קולט",
            "א, קולס":"אביטל קולט", "א. קולס":"אביטל קולט", "אי קולס": "אביטל קולט"
        }

        self.df['chairperson'] = self.df['chairperson'].replace(chairperson_map)
        enc = OneHotEncoder(sparse_output=False)
        encoded = enc.fit_transform(self.df[['chairperson']])
        self.df['chairperson_vector'] = list(encoded)
    
    def add_speaker_features(self):
        filtered = self.df[self.df['speaker_name'].apply(lambda x: 1 < len(x.split()) < 6 & len(x) > 3)]

        unique_names = filtered['speaker_name'].unique()
        unique_names = [name for name in unique_names if len(name.split()) > 2]

        name_map = {}
        oracle = pipeline('ner', model='dicta-il/dictabert-ner', aggregation_strategy='simple')
        oracle.tokenizer.backend_tokenizer.decoder = WordPiece()

        def extract_name(name, oracle):
            entities = oracle(name)
            extracted_name = ''
            for entity in entities:
                if entity['entity_group'] == 'PER':
                    extracted_name = entity['word']
                    break
            return extracted_name
        
        for name in tqdm(unique_names):
            name_map[name] = extract_name(name, oracle)
        
        filtered.loc[:, 'extracted_name'] = filtered.loc[:, 'speaker_name'].apply(lambda x: name_map[x] if x in name_map else x)
        filtered['extracted_name'] = filtered['extracted_name'].str.replace('[^\u0590-\u05FF ]', '', regex=True)
        filtered['extracted_name'] = filtered['extracted_name'].str.strip()

        enc = OneHotEncoder(sparse_output=False)
        encoded = enc.fit_transform(filtered[['extracted_name']])

        filtered['speaker_vector'] = list(encoded)
        self.df = filtered
    
    def add_committee_features(self):
        enc = OneHotEncoder(sparse_output=False)
        encoded = enc.fit_transform(self.df[['committee_name']])
        self.df['committee_vector'] = list(encoded)


if __name__ == '__main__':
    data_path = '/home/nogaschw/Call-to-order/Data2/base_data.csv'
    save_path = '/home/nogaschw/Call-to-order/Data2/base_data_features1030.csv'
    fe = FeatureExtractor(data_path)
