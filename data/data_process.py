import pandas as pd
import numpy as np
import requests
import lxml.html
import pickle
import json
from pprint import pprint
import os
from IPython.core.display import display, HTML
import html2text 
import re
from helpers import constants, utils, vocabulary
import nltk.data

nltk.download('punkt')

df = pd.DataFrame(columns=['date_filed', 'year_filed', 'name_first', 'name_last', 'political_party', 'text'])

for filename in os.listdir('./opinions'):
    if filename.endswith(".json"): 
        with open('./opinions/' + filename) as f_o:
            data_o = json.load(f_o)
            author = data_o.get('author')
            cluster = data_o.get('cluster')
            id = filename.split('.')[0]
            
            if author != None:
                author_id = author.split('/')[-2]
                cluster_id = cluster.split('/')[-2]
                
                with open('./people/' + str(author_id) + '.json') as f_a:
                    with open('./clusters/' + str(cluster_id) + '.json') as f_c:
                        data_a = json.load(f_a)
                        data_c = json.load(f_c)
                        
                        if data_o.get('html_lawbox') != None:
                            my_text = data_o.get('html_lawbox')
                        elif data_o.get('html') != None:
                            my_text = data_o.get('html')
                        elif data_o.get('html_columbia') != None:
                            my_text = data_o.get('html_columbia')
                        elif data_o.get('plain_text') != None:
                            my_text = data_o.get('plain_text')
                        else:
                            my_text = None
                            
                        if my_text != None and my_text != "":
                            df.loc[id, 'date_filed'] = data_c.get('date_filed')
                            df.loc[id, 'year_filed'] = df.loc[id, 'date_filed'][0:4]
                            df.loc[id, 'name_first'] = data_a.get('name_first')
                            df.loc[id, 'name_last'] = data_a.get('name_last')

                            party_data = data_a.get('political_affiliations')
                            if len(party_data) == 1:
                                df.loc[id, 'political_party'] = party_data[0].get('political_party')
                            else:
                                for i in range(len(party_data)):
                                    date_end = party_data[i].get('date_end')
                                    date_start = party_data[i].get('date_start')

                                    if date_end == None:
                                        date_end = '2100-01-01'

                                    if date_start == None:
                                        date_start = '1600-01-01'

                                    if date_start <= df.loc[id, 'date_filed'] and df.loc[id, 'date_filed'] < date_end:
                                        df.loc[id, 'political_party'] = party_data[i].get('political_party')
                            
                            doc = lxml.html.fromstring(my_text).text_content().split('\n')
                            sentences = nltk.sent_tokenize("".join(doc))
                            text = "<s> " + " <s> ".join(sentences)

                            df.loc[id, 'text'] = text
                            
                            if len(df) % 100 == 0:
                                print(len(df))
        continue
    else:
        continue
        
# Pull some processing steps from utils file
def canonicalize_digits(word):
    if any([c.isalpha() for c in word]): return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "") # remove thousands separator
    return word

def canonicalize_word(word, wordset=None, digits=True):
    word = word.lower()
    if digits:
        if (wordset != None) and (word in wordset): return word
        word = canonicalize_digits(word) # try to canonicalize numbers
    if (wordset == None) or (word in wordset):
        return word
    else:
        return constants.UNK_TOKEN

def canonicalize_words(words, **kw):
    return [canonicalize_word(word, **kw) for word in words]

df['tokens'] = df.text.apply(lambda x: canonicalize_words(x.split()))
df = df.drop(['text'], axis=1)

pickle.dump(df, open("data.p", "wb"))