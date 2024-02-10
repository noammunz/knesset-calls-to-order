import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import re
from my_parser import Parser
from collections import defaultdict, Counter
import utils

# a class for preprocessing the data

class Preprocessing:
    def __init__(self, data_path, save_path):
        self.data_path = data_path
        self.save_path = save_path

        data = self.load_data()
        data = self.unify_committees(data, 'unified_committee_name')
        data = self.add_call_to_order_label(data, 'call_to_order')
        self.save_data(data, 'preprocessed_data.csv')
    
    def load_data(self):
        data = pd.read_csv(self.data_path)
        return data
    
    def save_data(self, data, name):
        data.to_csv(self.save_path + name, index=False, encoding='utf-8-sig')
    
    def unify_committees(self, data, new_col_name):
        committees = data['committee_name'].unique()
        committees = [c for c in committees if c is 'ועד' in c]

        committee_groups = defaultdict(list)
        for c in committees:
            curr_group = []
            for c2 in committees:
                if c == c2:
                    continue
                levenshtein_score = utils.normalized_levenshtein(c, c2)
                if levenshtein_score < 0.3:
                    curr_group.append((c2, levenshtein_score))
            committee_groups[c] = curr_group
        
        committee_mapping = {}
        for name, similar_names in tqdm(committee_groups.items()):
            if name not in committee_mapping:
                chosen_name = name
            else:
                chosen_name = committee_mapping[name][0]
            
            for similar_name, score in similar_names:
                if similar_name not in committee_mapping:
                    committee_mapping[similar_name] = (chosen_name, score)
                else:
                    if committee_mapping[similar_name][1] > score:
                        committee_mapping[similar_name] = (chosen_name, score)
        
        data[new_col_name] = data['committee_name'].apply(lambda x: committee_mapping[x][0] if x in committee_mapping else x)
        return data
    
    def add_call_to_order_label(self, data, new_col_name):
        calls_to_order = ['אני קורא אותך לסדר', 'אני קוראת אותך לסדר', 'זאת אזהרה אחרונה',
            'קריאה לסדר', 'קריאת ראשונה לסדר', 'אנא, השתק והקשב',
            'אני מבקש ממך לשבת', 'אני מבקשת ממך לשבת', 'אני דורש שקט',
            'אני דורשת שקט', 'זו קריאה לשקט', 'אתה מתבקש להתיישב',
            'את מתבקשת להתיישב', 'אני פוסק את הדיון', 'אני פוסקת את הדיון',
            'נא להשתתק', 'נא לחדול מהרעש', 'הפסקת הדיון עכשיו',
            'אני מזהיר אותך לאחרונה', 'אני מזהירה אותך לאחרונה', 'קורא אותך',
            'קוראת אותך', 'סדר נא', 'שקט בבקשה',
            'סדר בבקשה',]

        data[new_col_name] = data['conversation'].apply(lambda x: 1 if any([call in x for call in calls_to_order]) else 0)
        return data