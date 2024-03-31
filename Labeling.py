import utils
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, BertModel

class Labeling:
    def __init__(self, path_data, save_path):
        self.call_to_order = ['אני קורא אותך לסדר', 'אני קוראת אותך לסדר', 'זאת אזהרה אחרונה',
                'קריאה לסדר', 'קריאת ראשונה לסדר', 'אני מזהיר אותך', 'קורא אותך', 
                'קוראת אותך', 'קורא לך לסדר', 'קוראת לך לסדר']
        self.data = utils.load_data(path_data)
        self.data.fillna('', inplace=True)
        self.data['committee_name'] = self.data['committee_name'].str.strip()
        self.data['chairperson'] = self.data['chairperson'].str.strip()
        self.data['speaker_name'] = self.data['speaker_name'].str.strip()
        self.add_call_to_order()
        self.add_modaration_calls()
        for i in range(len(self.data)):
            if self.data.at[i, 'call_to_order'] == 1:
                j = 1
                while i > j and self.data.at[i, 'speaker_name'] == self.data.at[i - j, 'chairperson'] and self.data.at[i - j, 'session_id'] == self.data.at[i, 'session_id']:
                    j += 1
                if self.data.at[i, 'speaker_name'] != self.data.at[i - j, 'chairperson'] and self.data.at[i - j, 'session_id'] == self.data.at[i, 'session_id']:
                    self.data.at[i - j, 'call_to_order'] = 1
                self.data.at[i, 'call_to_order'] = 0
        print('Labeling was done')
        self.data = self.data[['committee_name', 'session_id', 'chairperson', 'speaker_name', 'conversation', 'call_to_order']]
        utils.save_data(self.data, save_path)
        
    def add_call_to_order(self):
        """
        Add call to order to the data by regex
        """
        self.data['call_to_order'] = self.data['conversation'].apply(lambda x: 1 if any([call in x for call in self.call_to_order]) else 0)
        print(f"Number of call to order: {self.data['call_to_order'].sum()}")

    def _get_cls_token(self, model, tokenizer, texts, device):
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

    def add_modaration_calls(self):
        model_name = "onlplab/alephbert-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Get CLS tokens
        cls_tokens = []
        batch_size = 32
        for i in range(0, len(self.data), batch_size):
            curr_cls_tokens = self._get_cls_token(model, tokenizer, self.data['conversation'][i:i+batch_size].tolist(), device)
            cls_tokens.append(curr_cls_tokens.cpu().numpy())
        self.data['cls_token'] = list(np.concatenate(cls_tokens))
        print("CLS tokens were extracted")

        # Get cosine similarities
        cto_cls_tokens = self._get_cls_token(model, tokenizer, self.call_to_order, device).cpu().numpy()
        all_cls_tokens = np.array(self.data['cls_token'].tolist())
        cosine_similarities = cosine_similarity(all_cls_tokens, cto_cls_tokens)
        print("Cosine similarities were calculated")

        # Find top similarities
        top_similarities = []
        batch_size = 1024
        for i in range(0, len(self.data), batch_size):
            curr_similarities = cosine_similarities[i:i+batch_size]
            curr_top_similarity = np.max(curr_similarities, axis=1)
            top_similarities.append(curr_top_similarity)
        self.data['top_similarities'] = np.concatenate(top_similarities)
        self.data.drop(columns='cls_token', inplace=True)
        print("Top similarities were found")
        
        # Add moderation calls
        for idx, row in self.data.iterrows():
            curr_sim = row['top_similarities']
            chairperson_speaking = row['chairperson'] == row['speaker_name']
            if curr_sim > 0.83 and chairperson_speaking:
                self.data.at[idx, 'call_to_order'] = 1
        print(f"Number of call to order after moderation calls: {self.data['call_to_order'].sum()}")