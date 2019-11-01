#!/usr/bin/env python
# coding: utf-8

# ## KeyPhrase extraction
# * Use Inspec dataset (abstracts) - [train](https://github.com/boudinfl/ake-datasets/blob/master/datasets/Inspec/train/) dataset for BM25 vectorization
# * Use Inspec dataset (abstracts) - [test](https://github.com/boudinfl/ake-datasets/blob/master/datasets/Inspec/test/) for inference

# In[1]:


import re
import operator
import json
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from nltk import ngrams
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from xml.etree import ElementTree
from collections import Counter

from tqdm import tqdm_notebook as tqdm

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 500)


# In[2]:


from nltk.stem.snowball import SnowballStemmer
sno = SnowballStemmer('english')


# In[3]:


def read(directory):
    docs = {}
    for doc_path in tqdm(glob(f'{directory}/*.xml')):
        doc = ElementTree.parse(doc_path)
        sentences = []
        for sentence in doc.find('document').find('sentences').findall('sentence'):
            sentences.append(' '.join([token.find('lemma').text.lower() + '~' + token.find('POS').text
                                       for token in sentence.find('tokens').findall('token')]))

        docs[doc_path.split('/')[-1].split('.')[0]] = '\n'.join(sentences)
    return docs


# In[4]:


train_sentences = read('ake-datasets/datasets/Inspec/train')
test_sentences = read('ake-datasets/datasets/Inspec/test')
len(train_sentences), len(test_sentences)


# In[5]:


pattern = re.compile(r'(((\w+~JJ)* (\w+~NN)+ (\w+~IN))?(\w+~JJ)+ (\w+~NN)+)+')


# In[6]:


train_candidates = {doc_id: [candidate[0] for candidate in re.findall(pattern, doc)] for doc_id, doc in train_sentences.items()}
train_candidates = {doc_id: [' '.join([w.split('~')[0] for w in candidate.split()]) for candidate in candidates] for doc_id, candidates in train_candidates.items()}
train_sentences = {doc_id: ' '.join([w.split('~')[0] for w in sentences.split()]) for doc_id, sentences in train_sentences.items()}
train_frequencies = {doc_id: Counter(
                                [' '.join(gram) for gram in ngrams(doc.split(), 1)] + \
                                [' '.join(gram) for gram in ngrams(doc.split(), 2)] + \
                                [' '.join(gram) for gram in ngrams(doc.split(), 3)])
                    for doc_id, doc in train_sentences.items()}


# In[ ]:





# In[7]:


test_candidates = {doc_id: [candidate[0] for candidate in re.findall(pattern, doc)] for doc_id, doc in test_sentences.items()}
test_candidates = {doc_id: [' '.join([w.split('~')[0] for w in candidate.split()]) for candidate in candidates] for doc_id, candidates in test_candidates.items()}
test_sentences = {doc_id: ' '.join([w.split('~')[0] for w in sentences.split()]) for doc_id, sentences in test_sentences.items()}
test_frequencies = {doc_id: Counter(
                                [' '.join(gram) for gram in ngrams(doc.split(), 1)] + \
                                [' '.join(gram) for gram in ngrams(doc.split(), 2)] + \
                                [' '.join(gram) for gram in ngrams(doc.split(), 3)])
                    for doc_id, doc in test_sentences.items()}


# In[8]:


vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 3))
trainvec = vectorizer.fit_transform(train_sentences.values())
feature_names = vectorizer.get_feature_names()


# In[9]:


with open('ake-datasets/datasets/Inspec/references/test.uncontr.json', 'r') as f:
    target = json.load(f)
    target = {doc_name: [k[0] for k in keyphrases] for doc_name, keyphrases in target.items()}


# In[10]:


target['193']


# In[11]:


test_candidates['193']


# In[21]:


from math import log
def score(t, d, k1=1.2, b=0.75):
    """
    :param t: term
    :param d: document-id in test dataset
    
    ftd = f(t, d): term frequency
    avgdl = mean([len(doc) for doc in train])
    N = len(train)
    nt = n(t) = sum(1 for doc in train if t in doc)
    """
    N = len(train_frequencies)
    nt = sum(1 for doc in train_frequencies.values() if t in doc)
    avgdl = np.mean([sum(frequencies.values()) for frequencies in train_frequencies.values()])
    ftd = 1. * test_frequencies[d][t] / test_frequencies[d].most_common(1)[0][1]
    ld = sum(test_frequencies[d].values())
    
    tf = (ftd * (k1 + 1)) / (ftd + k1 * (1 - b + b * ld / avgdl))
    idf = log((N - nt + 0.5) / (nt + 0.5))
    return tf * idf


# In[22]:


score('out-of-print material', '193')


# In[23]:


def extract_keyphrases(doc_id, nb_keywords=5):
    scores = {candidate: score(candidate, doc_id) for candidate in test_candidates[doc_id]}
    scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)[:nb_keywords]
    return [keyphrase for keyphrase, score in scores]


# In[24]:


predictions = {doc_id: extract_keyphrases(doc_id, nb_keywords=5) for doc_id, doc in tqdm(test_sentences.items())}


# In[25]:


predictions = {doc_id: [sno.stem(candidate) for candidate in candidates] for doc_id, candidates in predictions.items()}
target = {doc_id: [sno.stem(candidate) for candidate in candidates] for doc_id, candidates in target.items()}


# In[26]:


predictions['193'], target['193']


# In[29]:


def avg_precisoin(pred, targ):
    res, nb_correct = 0, 0
    for i, p in enumerate(pred):
        if p in targ:
            nb_correct += 1
        res += nb_correct / (i + 1)
    return 1. / len(targ) * res


# In[30]:


results = []
for doc_id in sorted(predictions.keys()):
    p = set(predictions[doc_id])
    t = set(target[doc_id])
    at_5 = set(target[doc_id][:5])

    # We always predict 5 keywords
    precision = 0 if len(p) == 0 else len(p.intersection(t)) / len(p)
    recall = 0 if len(t) == 0 else len(p.intersection(t)) / len(t)
    results.append({
        'doc_id':      doc_id,
        'precision':   precision,
        'recall':      recall,
        'f1':          0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall),
        'precision@5': len(p.intersection(at_5)) / 5.,
        'av_prec':     avg_precisoin(p, t)
    })

results = pd.DataFrame(results)
results.set_index('doc_id', inplace=True)

print('Precision: {:.2f} Recall: {:.2f} F1: {:.2f}   precision@5: {:.2f}  MAP: {:.2f}'.format(
    results["precision"].mean(),
    results["recall"].mean(),
    results["f1"].mean(),
    results["precision@5"].mean(),
    results["av_prec"].mean()
))
print('--------------Mean-------------')
results


# In[ ]:





# In[ ]:





# In[ ]:




