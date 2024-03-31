import re
import os
import requests
from bs4 import BeautifulSoup

def get_urls():
    """
    get all the urls of the text files from the oknesset website.
    """
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

def filter_by_call_to_order(text):
    """
    check if the text contains a call to order.
    """
    calls_to_order = ['אני קורא אותך לסדר', 'אני קוראת אותך לסדר', 'זאת אזהרה אחרונה',
                'קריאה לסדר', 'קריאת ראשונה לסדר', 'אני מזהיר אותך', 'קורא אותך', 
                'קוראת אותך', 'קורא לך לסדר', 'קוראת לך לסדר']

    for call in calls_to_order:
        if call in text:
            return True
    return False

def get_text(url, dir='text'):
    """
    get the text from the url and save it in a file in the dir. Only if the text contains a call to order.
    """
    response = requests.get(url)
    if response.status_code == 200:
        response.encoding = 'utf-8-sig'
        if not os.path.exists(dir):
            os.makedirs(dir)
        text = response.text
        text = text.replace('”', '"').replace('“', '"').replace('״', '"')
        if filter_by_call_to_order(text):
            text = re.split('\n', text)
            with open(f'{dir}/{url.split("/")[-1]}', 'x', encoding='utf-8-sig') as file:
                file.write('\n'.join(text))
        else:
            print(f"Not found call to order in {url}.")

    else:
        print(f"Failed to retrieve content from {url}. Status code: {response.status_code}")
        return None  