import os
import re
import utils
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from collections import defaultdict
from tokenizers.decoders import WordPiece

class Preprocessing:
    """
    This class is responsible for preprocessing the data.
    """
    def __init__(self, data_path, save_path, base_path="", filter_data=True, create_new=False):
        self.data_path = os.path.join(base_path, data_path)
        self.save_path = os.path.join(base_path, save_path)

        # if save path already exists, don't create new data
        if os.path.exists(self.save_path) and not create_new:
            self.data_path = self.save_path
            self.data = utils.load_data(data_path)
            return
        
        self.data = utils.load_data(data_path)
        self.data = self.data[['committee_name', 'session_id', 'chairperson', 'speaker_name', 'conversation']]
        self.data.dropna(inplace=True)
        self.unify_committees()
        self.accurate_conv()
        self.data.sort_values(by=['session_id'], inplace=True)
        utils.save_data(self.data, self.save_path)
    
    def unify_committees(self):
        """
        This function unifies similar committee names.
        """
        committees = self.data['committee_name'].unique()
        committees = [c for c in committees if 'ועד' in c]

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
        
        self.data['committee_name'] = self.data['committee_name'].apply(lambda x: committee_mapping[x][0] if x in committee_mapping else x)

    def accurate_conv(self):
        """
        make sure long conversations are split into smaller ones
        """
        oracle = pipeline('ner', model='dicta-il/dictabert-ner', aggregation_strategy='simple')
        oracle.tokenizer.backend_tokenizer.decoder = WordPiece()
        remove_from_df = []
        add_to_df = {}
        for inx, conv in enumerate(self.data['conversation']):
            if len(conv.split()) > 40:
                entities = oracle(conv)
                speakers = [speaker for speaker in entities if speaker['entity_group'] == 'PER']
                if len(speakers) == 0:
                    continue
                remove_from_df.append(inx)
                prev = speakers.pop(0)
                if len(speakers) == 0:
                    if add_to_df.get(inx) is None:
                        add_to_df[inx] = []
                    add_to_df[inx].append((prev['word'], conv))
                for speaker in speakers:
                    speaker_turn = conv[prev['end']:speaker['start']]
                    speaker_turn = re.sub(r'\b(?:יו"ר|יור|היו"ר)\b', '', speaker_turn)
                    if add_to_df.get(inx) is None:
                        add_to_df[inx] = []
                    add_to_df[inx].append((prev['word'], speaker_turn))
                    prev = speaker
            print(f"Done {inx}/{len(self.data)}", end='\r')
        print("\n")
        i = 0
        for inx in remove_from_df:
            print(f"Replace {inx}/{len(remove_from_df)}", end='\r')
            row = self.data.iloc[inx - i]
            self.data.drop(self.data.index[inx - i], inplace=True)
            i += 1
            for speaker, turn in add_to_df[inx]:
                new_index = self.data.index.max() + 1
                new_row = {'committee_name':row[0], 'session_id':row[1], 'chairperson':row[2], 'speaker_name': speaker, 'conversation': turn}
                self.data.loc[new_index] = new_row