import os
import pandas as pd

class WindowsData:
    """
    This class is responsible for creating windows of text and speaker names.
    """
    def __init__(self, data, save_path, windows_size=1):
        self.windows_size = windows_size
        df = pd.DataFrame(data)
        if windows_size == 1:
            self.data = df
            return
        self.data = df.groupby('session_id').apply(self.sliding_window_concat).reset_index(drop=True)
        path_part, extension_part = save_path.rsplit('.', 1)
        new_filepath = f"{path_part}{self.windows_size}.{extension_part}"
        self.data.to_csv(new_filepath, index=False, encoding='utf-8-sig')

    def sliding_window_concat(self, group):
        group['speaker_name'] = group['speaker_name'].astype(str)
        group['conversation'] = group['conversation'].astype(str)
        
        # This function generates sliding windows and concatenates within each window
        speaker_windows = [group['speaker_name'].iloc[i+self.windows_size-1] for i in range(len(group) - self.windows_size + 1)]
        conversation_windows = ['#משפט#'.join(group['conversation'].iloc[i:i+self.windows_size]) for i in range(len(group) - self.windows_size + 1)]
        call_to_order = [group['call_to_order'].iloc[i+self.windows_size-1] for i in range(len(group) - self.windows_size + 1)]

        result_df = pd.DataFrame({
            'committee_name': group['committee_name'].iloc[-1],
            'session_id': group['session_id'].iloc[-1],
            'chairperson': group['chairperson'].iloc[-1],
            'speaker_name': speaker_windows, 
            'conversation': conversation_windows,
            'call_to_order': call_to_order
        })
        return result_df