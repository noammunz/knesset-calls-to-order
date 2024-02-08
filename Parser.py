import re
import csv
import datetime
import requests
from bs4 import BeautifulSoup

class Parser:
    def __init__(self):
        with open('data.csv', 'w', encoding='utf-8-sig', newline='') as csvfile:
            start_time = datetime.datetime.now()
            print(f"Start time: {start_time.strftime('%H:%M:%S')}")
            writer = csv.writer(csvfile)
            writer.writerow(['number', 'committee_name', 'subject', 'start_conver', 'speaker_name', 'conversation'])
            self.urls = self.get_urls()
            counter = 1
            for url in self.urls:
                text = self.get_protocol(url)
                if text is None or text == []:
                    continue
                committee_name = self.get_committee_name(text)
                subject = self.get_subject(text)
                start_conver = self.get_start_conver(text)
                conversation = self.create_conversation(text, start_conver)
                for speaker, speaker_text in conversation:
                    writer.writerow([counter, committee_name, subject, start_conver, speaker, speaker_text])
                counter += 1
                if counter % 100 == 0:
                    time_elapsed = datetime.datetime.now() - start_time
                    print(f"Finished {counter/len(self.urls)*100:.2f}%, time elapsed: {time_elapsed}")
            
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
    
    def get_protocol(self, url):
        print("Getting protocol from", url)
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
        for i in range(1, len(text)):
            if 'ועד' in text[i]:
                text[i] = re.sub(r'[^א-ת ]', '', text[i])
                return text[i]
            
    def get_subject(self, text):
        if not text[0].isdigit() and text[0] != 'PAGE':
            return text[0]
        
    def get_start_conver(self, text):
        count = 0
        for row in text:
            if 'יו"ר' in row and ':' in row and count > 20:
                return count
            count += 1
        return 20

    def create_conversation(self, text, line_number):
        print("Creating conversation")
        conversation = []
        if len(text) < line_number + 1:
            conversation.append((None, text))
            return conversation
        name = text[line_number]
        conversation_text = text[line_number + 1:]
        text = ""
        for row in conversation_text:
            if row[-1] == ':':
                conversation.append((name.replace('\t', ''), text.replace('\t', '')))
                name = row
                text = ""
            else:
                text += row
        return conversation  