import re
import pandas as pd

def normalized_levenshtein(s1, s2):
        if len(s1) < len(s2):
            return normalized_levenshtein(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1] / max(len(s1), len(s2))

def load_data(data_path):
    return pd.read_csv(data_path)

def save_data(df, save_path):
    df.to_csv(save_path, index=False, encoding='utf-8-sig')