import os
import re
import csv
import datetime
from transformers import pipeline
from tokenizers.decoders import WordPiece


class Parser:
    """
    A class for parsing the text files from the Knesset website and extracting the relevant information
    """
    def __init__(self, start_index=0, texts_path='text', log_path=r'log.txt'):
        self.question_answerer = pipeline('question-answering', model="tdklab/hebert-finetuned-hebrew-squad")
        self.log_path = log_path

        with open('my_data.csv', 'w', encoding='utf-8-sig', newline='') as csvfile:
            print('STARTING')
            self.start_time = datetime.datetime.now()
            writer = csv.writer(csvfile)
            writer.writerow(['committee_name', 'session_id', 'format', 'chairperson', 'text_id', 'speaker_name', 'conversation'])

            self.skipped = 0
            iters = 0
            self.nums_files = len(os.listdir(texts_path))
            for file_name in os.listdir(texts_path):
                session_id = int(file_name.split('.')[0])
                with open(os.path.join(texts_path, file_name), 'r', encoding='utf-8-sig') as file:
                    text = file.read().split('\n')
                self._counter(iters)
                if self._check_error(text, 'text', session_id):
                    continue

                committee_name = self.get_committee_name(text)
                chairperson = self.get_chairperson(text)
                start_index, format = self.get_start_index(text, chairperson)
                if self._check_error(committee_name, 'committee', session_id) or self._check_error(chairperson, 'chairperson', session_id) or self._check_error(start_index, 'start index', session_id):
                    continue

                conversation = self.get_conversation(text, start_index, format, chairperson)
                for text_id, (speaker, speaker_text) in enumerate(conversation):
                    writer.writerow([committee_name, session_id, format,  chairperson, text_id, speaker, speaker_text])
                iters += 1
                
            with open(log_path, 'a') as file:
                time_elapsed = str(datetime.datetime.now() - self.start_time)[:7]
                file.write(f'Finished, skipped: {self.skipped}, time_elapsed: {time_elapsed}')
                
    def _counter(self, iters):
        if iters % 150 == 0:
            time_elapsed = str(datetime.datetime.now() - self.start_time)[:7]
            print(f'Finished {iters} out of {self.nums_files} --- {iters/self.nums_files*100:.2f}%, time elapsed: {time_elapsed},  time: {datetime.datetime.now()}\n')
            with open(self.log_path, 'a') as file:
                file.write(f'Finished {iters} out of {self.nums_files} --- {iters/self.nums_files*100:.2f}%, time elapsed: {time_elapsed},  time: {datetime.datetime.now()}\n')

    def _check_error(self, info, type, file_name):
        if info is None or info == [] or info == '':
            self.skipped += 1
            print(f'Failed to get {type}: {file_name}\n')
            with open(self.log_path, 'a') as file:
                file.write(f'\nFailed to get {type}: {file_name}\n\n')
            return True
        return False
       
    def get_committee_name(self, text):
        context = ' '.join(text[3:30])
        context = re.sub(r'[^א-ת ]', '', context)
        question = 'מה שמה של הוועדה?'
        question2 = 'מה שמה של הישיבת?'
        if context == '':
            return None
        answer = self.question_answerer(question=question, context=context)["answer"]
        if "פרוטוקול" in answer or "מושב" in answer or len(answer.split()) == 1:
            answer = self.question_answerer(question=question2, context=context)["answer"]
        match = re.search(r'\bמ?יום\b', answer)
        if match:
            index = match.start()
            answer = answer[:index]
        
        answer = re.sub(r'[^א-ת ]', '', answer).strip()
        return answer
    
    def _number_appearnece(self, text, chairperson):
        number = 0
        for row in text:
            if chairperson in row:
                number += 1
            if number > 3:
                return True
        return False

    def get_chairperson(self, text):
        question = 'מה שמו של יו"ר הוועדה?'
        
        for row in text:
            if ('יו"ר' in row or '<< יור >>' in row) and len(row.split()) > 2 and len([word for word in row.split() if word.isalnum()]) < 7:
                context = row
                answerq = self.question_answerer(question=question, context=context)
                if answerq['score'] > 0.5:
                    answer = answerq['answer']
                    match = re.search(r'(.*?)\b(היו"ר)\b(.*)', answer)
                    if match:
                        answer = match.group(1) if len(match.group(1)) > len(match.group(3)) else match.group(3)

                    if len(answer) > 3:
                        answer = ' '.join(answer.split()[0:2])
                    answer = answer.strip()
                    if self._number_appearnece(text, answer):
                        return answer
        return None
    
    def get_start_index(self, text, chairperson):
        start_index = None
        if chairperson is None:
            return (None, None)
            
        if '<< יור >>' in ' '.join(text):
            for i, row in enumerate(text):
                if '<< יור >>' in row:
                    return (i, 0)

        for i, row in enumerate(text):
            if chairperson in row and row.strip()[-1] == ':':
                return (i, 1)

        next_time = 0
        for i, row in enumerate(text):
            if next_time and row != '' and len(row) > 5:
                if "לפתוח" in row or "פותח" in row:
                    return (next_time, 3)
                else:
                    next_time = 0  
            if chairperson in row:
                if "לפתוח" in row or "פותח" in row:
                    return (i, 4)
                next_time = i
                
        if start_index is None: # second format
            for i, row in enumerate(text[25:]):
                if chairperson in row and ':' in row:
                    return (i + 25, 2)
        return (start_index, None)

    def get_conversation(self, text, start_index, format, chairperson):
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
            else:
                return [(chairperson, ' '.join(text[start_index:]))]
        if utterance:
            conversation.append((current_speaker, '. '.join(utterance)))

        return conversation