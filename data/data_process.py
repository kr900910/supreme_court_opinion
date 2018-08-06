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
import nltk.data
import gensim
import string
import itertools

# Load pre-trained word2vec model
model = gensim.models.KeyedVectors.load_word2vec_format('../model/GoogleNews-vectors-negative300.bin', binary=True)
vocab = set(model.wv.vocab)
model = None
exclude = set(string.punctuation)

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Pull some processing steps from utils file
def canonicalize_digits(word):
    if any([c.isalpha() for c in word]): return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "") # remove thousands separator
    return word

def canonicalize_word(word, digits=True):
    word = word.lower()
    if digits:
        if word in vocab: return word
        word = canonicalize_digits(word) # try to canonicalize numbers
    if word in vocab:
        return word
    else:
        return u"<unk>"

def canonicalize_words(words, **kw):
    return [canonicalize_word(word, **kw) for word in words]

# The function looks at tags for words in sentences
# and only return the sentences which have both noun and verb
def sent_helper(s):
    tokens = nltk.word_tokenize(s)
    tags = nltk.pos_tag(tokens)
    verb = [1 if 'VB' in w[1] else 0 for w in tags]
    noun = [1 if 'NN' in w[1] else 0 for w in tags]
    pronoun = [1 if 'PRP' in w[1] else 0 for w in tags]
    if sum(verb) > 0 and (sum(noun) > 0 or sum(pronoun) > 0):
        new_s = ''.join(ch for ch in s if ch not in exclude)
        tokens = canonicalize_words(new_s.split())
        return tokens + ['</s>']
    else:
        return []

df = pd.DataFrame(columns=['date_filed', 'year_filed', 'name_first', 'name_last', 'political_party', 'text'])

doc_id = 0

for filename in os.listdir('./opinions'):
    if filename.endswith(".json"): 
        with open('./opinions/' + filename) as f_o:
            data_o = json.load(f_o)
            author = data_o.get('author')
            cluster = data_o.get('cluster')
            
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
                            date_filed = data_c.get('date_filed')
                            year_filed = date_filed[0:4]
                            name_first = data_a.get('name_first')
                            name_last = data_a.get('name_last')
                            
                            party_data = data_a.get('political_affiliations')
                            if len(party_data) == 1:
                                political_party = party_data[0].get('political_party')
                            else:
                                for i in range(len(party_data)):
                                    date_end = party_data[i].get('date_end')
                                    date_start = party_data[i].get('date_start')

                                    if date_end == None:
                                        date_end = '2100-01-01'

                                    if date_start == None:
                                        date_start = '1600-01-01'

                                    if date_start <= date_filed and date_filed < date_end:
                                        political_party = party_data[i].get('political_party')

                            doc = lxml.html.fromstring(my_text).text_content().split('\n')
                            sentences = nltk.sent_tokenize("".join(doc))
                            
                            df.loc[doc_id, 'date_filed'] = date_filed
                            df.loc[doc_id, 'year_filed'] = year_filed
                            df.loc[doc_id, 'name_first'] = name_first
                            df.loc[doc_id, 'name_last'] = name_last
                            df.loc[doc_id, 'political_party'] = political_party
                            df.loc[doc_id, 'text'] = list(itertools.chain.from_iterable([sent_helper(s) for s in sentences]))
                            
                            if doc_id % 100 == 0:
                                print(doc_id)
                            
                            doc_id += 1         
        continue
    else:
        continue

pickle.dump(df, open("data.p", "wb"))
