import re
import csv
import datetime
import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForQuestionAnswering, pipeline
import torch
import os

class Parser:
    def __init__(self, start_index=0, end_index=1000000):
        self.question_answerer = pipeline('question-answering', model="tdklab/hebert-finetuned-hebrew-squad")

        log_path = r'G:\School\Year 4\nlp\prject\log.txt'
        urls_path = r'G:\School\Year 4\nlp\prject\urls.txt'
        texts_path = r'G:\school_local\texts'
        with open('my_data.csv', 'w', encoding='utf-8-sig', newline='') as csvfile:
            print(f'STARTING')
            start_time = datetime.datetime.now()
            writer = csv.writer(csvfile)
            writer.writerow(['committee_id', 'committee_name', 'session_id', 'format', 'chairperson', 'text_id', 'speaker_name', 'conversation'])

            committee_id = 0
            skipped = 0
            iters = 0
            nums_files = len(os.listdir(texts_path))
            for file_name in os.listdir(texts_path):
                session_id = int(file_name.split('.')[0])
                with open(os.path.join(texts_path, file_name), 'r', encoding='utf-8-sig') as file:
                    text = file.read().split('\n')

                if iters % 150 == 0:
                    time_elapsed = str(datetime.datetime.now() - start_time)[:7]
                    print(f'Finished {iters} out of {nums_files} --- {iters/nums_files*100:.2f}%, time elapsed: {time_elapsed},  time: {datetime.datetime.now()}\n')
                    with open(log_path, 'a') as file:
                        file.write(f'Finished {iters} out of {nums_files} --- {iters/nums_files*100:.2f}%, time elapsed: {time_elapsed},  time: {datetime.datetime.now()}\n')
                
                session_id += 1
                if text is None or text == []:
                    skipped += 1
                    print(f'Failed to get text: {session_id}\n')
                    with open(log_path, 'a') as file:
                        file.write(f'\nFailed to get text: {session_id}\n\n')
                    continue

                committee_name = self.get_committee_name(text)
                if committee_name is None:
                    skipped += 1
                    print(f'Failed to get committee: {session_id}\n')
                    with open(log_path, 'a') as file:
                        file.write(f'\nFailed to get committee: {session_id}\n\n')
                    continue

                chairperson = self.get_chairperson(text)
                if chairperson is None:
                    skipped += 1
                    print(f'Failed to get chairperson: {session_id}\n')
                    with open(log_path, 'a') as file:
                        file.write(f'\nFailed to get chairperson: {session_id}\n\n')
                    continue

                start_index, format = self.get_start_index(text, chairperson)
                if start_index is None:
                    skipped += 1
                    print(f'Failed to get start index: {session_id}\n')
                    with open(log_path, 'a') as file:
                        file.write(f'\nFailed to get start index: {session_id}\n\n')
                    continue

                conversation = self.get_conversation(text, start_index, format)
                for text_id, (speaker, speaker_text) in enumerate(conversation):
                    writer.writerow([committee_id, committee_name, session_id + 1, format,  chairperson, text_id, speaker, speaker_text])
                
                iters += 1
                
            with open(log_path, 'a') as file:
                time_elapsed = str(datetime.datetime.now() - start_time)[:7]
                file.write(f'Finished, skipped: {skipped}, time_elapsed: {time_elapsed}')
                
    def get_urls(self):
        url = 'https://production.oknesset.org/pipelines/data/committees/meeting_protocols_text/files/'
        folder_urls = [url]
        urls = []

        while(len(folder_urls) > 0):
            curr_url = folder_urls.pop(0)
            response = requests.get(curr_url, allow_redirects=False)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                all_links = soup.find_all('a')
                file_names = [link.text for link in all_links if link.text]
                file_names = [file_name for file_name in file_names if file_name != '..']
                for file_name in file_names:
                    if file_name.endswith('/') and file_name != '../':
                        folder_urls.append(curr_url + file_name)
                    elif file_name.endswith('.txt'):
                        urls.append(curr_url + file_name)
            else:
                print(f"Failed to retrieve content from {url}. Status code: {response.status_code}")
        return urls
    
    def get_text(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            response.encoding = 'utf-8-sig'
            text = response.text
            text = text.replace('”', '"').replace('“', '"').replace('״', '"')
            text = re.split('\n', text)
            text = [row for row in text if row.strip() != '']
            return text
        else:
            print(f"Failed to retrieve content from {url}. Status code: {response.status_code}")
            return None
    
    def get_committee_name(self, text):
        context = ' '.join(text[1:10])
        question = 'מה שמה של הוועדה?'
        if context == '':
            return None
        answer = self.question_answerer(question=question, context=context)['answer']
        # handling common errors
        match = re.search(r'\bמ?יום\b', answer)
        if match:
            index = match.start()
            answer = answer[:index]
        
        answer = re.sub(r'[^א-ת ]', '', answer).strip()
        return answer

    def get_chairperson(self, text):
        chairperson_rows = []
        for row in text[::-1]:
            if ('יו"ר' in row and len(row) < 65) or ('<< יור >>' in row):
                chairperson_rows.append(row)
            if len(chairperson_rows) == 2:
                break
        
        context = ' '.join(chairperson_rows)
        if context == '':
            return None
        question = 'מה שמו של יו"ר הוועדה?'
        answer = self.question_answerer(question=question, context=context)['answer']

        # handling common errors
        if ':' in answer:
            index = answer.index(':')
            answer = answer[:index]

        match = re.search(r'\bהיו"ר\b', answer)
        if match:
            index = match.start()
            answer = answer[index + 5:]
        
        answer = answer.strip()
        return answer
    
    def get_start_index(self, text, chairperson):
        start_index = None
        if chairperson is None:
            return None
        
        if '<< יור >>' in ' '.join(text):
            # return index of first row with '<< יור >>'
            for i, row in enumerate(text):
                if '<< יור >>' in row:
                    return (i, 0)

        for i, row in enumerate(text):
            if chairperson in row and row.strip()[-1] == ':':
                return (i, 1)

        if start_index is None: # second format
            for i, row in enumerate(text[25:]):
                if chairperson in row and ':' in row:
                    return (i + 25, 2)
        
        return (None, None)

    def get_conversation(self, text, start_index, format):
        conversation = []
        current_speaker = None
        utterance = []

        for i in range(start_index, len(text)):
            line = text[i].strip()

            if format == 0:
                if '<<' in line and '>>' in line:
                    speaker_start_index = line.find('<<') + 2
                    speaker_end_index = line.find('>>')
                    speaker = line[speaker_start_index:speaker_end_index].strip()                
                        
                    name_start_index = line.find('>>', speaker_end_index) + 2
                    name_end_index = line.find(':', name_start_index)
                    name = line[name_start_index:name_end_index].strip()

                    if '(' in name:
                        name = name[:name.index('(')].strip()

                    match = re.search(r'\bהיו"ר\b', name)
                    if match:
                        index = match.start()
                        name = name[index + 5:]

                    if utterance:
                        conversation.append((current_speaker, '\n'.join(utterance)))
                        utterance = []

                    current_speaker = name
                else:
                    utterance.append(line)
            elif format == 1:
                if line.endswith(':'):
                    if utterance:
                        conversation.append((current_speaker, '. '.join(utterance)))
                        utterance = []
                    current_speaker = line[:-1]
                    match = re.search(r'\bהיו"ר\b', current_speaker)  
                    if match:
                        index = match.start()
                        current_speaker = current_speaker[index + 5:]
                else:
                    utterance.append(line)
            elif format == 2:
                if ':' in line:
                    speaker, text_content = line.split(':', 1)
                    if current_speaker != speaker:
                        if utterance:
                            conversation.append((current_speaker, '. '.join(utterance)))
                            utterance = []
                        current_speaker = speaker
                        match = re.search(r'\bהיו"ר\b', current_speaker)  
                        if match:
                            index = match.start()
                            current_speaker = current_speaker[index + 5:]
                    utterance.append(text_content.strip())
                else:
                    utterance.append(line)

        if utterance:
            conversation.append((current_speaker, '. '.join(utterance)))

        return conversation

if __name__ == '__main__':
    parser = Parser()