#!/usr/bin/env python
# coding: utf-8

# ## KeyPhrase extraction
# * Use Inspec dataset (abstracts) - [train](https://github.com/boudinfl/ake-datasets/blob/master/datasets/Inspec/train/) dataset for supervised training
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
from math import log

from tqdm import tqdm_notebook as tqdm

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 500)


from nltk.stem.snowball import SnowballStemmer
sno = SnowballStemmer('english')


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


train_sentences = read('ake-datasets/datasets/Inspec/train')
test_sentences = read('ake-datasets/datasets/Inspec/test')
len(train_sentences), len(test_sentences)


# ### Classification method
# * Select all unigrams, bigrams, trigrams from the doc
# * Create a pandas data frame with
#     * Features: TF, IDF, TF-IDF, BM25, length of the token
#     * Class: 1 if it's a target keyphrase 0 if not
# * Use xgboost for the classification

pattern = re.compile(r'(((\w+~JJ)* (\w+~NN)+ (\w+~IN))?(\w+~JJ)+ (\w+~NN)+)+')


train_candidates = {doc_id: [candidate[0] for candidate in re.findall(pattern, doc)] for doc_id, doc in train_sentences.items()}
train_candidates = {doc_id: [' '.join([w.split('~')[0] for w in candidate.split()]) for candidate in candidates] for doc_id, candidates in train_candidates.items()}
train_sentences = {doc_id: ' '.join([w.split('~')[0] for w in sentences.split()]) for doc_id, sentences in train_sentences.items()}
train_frequencies = {doc_id: Counter(
                                [' '.join(gram) for gram in ngrams(doc.split(), 1)] + \
                                [' '.join(gram) for gram in ngrams(doc.split(), 2)] + \
                                [' '.join(gram) for gram in ngrams(doc.split(), 3)])
                    for doc_id, doc in train_sentences.items()}

test_candidates = {doc_id: [candidate[0] for candidate in re.findall(pattern, doc)] for doc_id, doc in test_sentences.items()}
test_candidates = {doc_id: [' '.join([w.split('~')[0] for w in candidate.split()]) for candidate in candidates] for doc_id, candidates in test_candidates.items()}
test_sentences = {doc_id: ' '.join([w.split('~')[0] for w in sentences.split()]) for doc_id, sentences in test_sentences.items()}
test_frequencies = {doc_id: Counter(
                                [' '.join(gram) for gram in ngrams(doc.split(), 1)] + \
                                [' '.join(gram) for gram in ngrams(doc.split(), 2)] + \
                                [' '.join(gram) for gram in ngrams(doc.split(), 3)])
                    for doc_id, doc in test_sentences.items()}


def tf(d, t, frequencies):
    return 1. * frequencies[d][t] / frequencies[d].most_common(1)[0][1]

def idf(t, frequencies):
    N = 1. * len(frequencies)
    nt = sum(1 for doc in frequencies.values() if t in doc)
    return log(N / nt) if N != 0 and nt != 0 else 0

dls = {}
def bm25(t, d, frequencies, background_frequencies, k1=1.2, b=0.75):
    """
    :param t: term
    :param d: document-id in test dataset
    
    ftd = f(t, d): term frequency
    avgdl = mean([len(doc) for doc in train])
    N = len(train)
    nt = n(t) = sum(1 for doc in train if t in doc)
    """
    N = len(background_frequencies)
    nt = sum(1 for doc in background_frequencies.values() if t in doc)
    # Dangerous but works for our train/test split
    if len(background_frequencies) not in dls:
        dls[len(background_frequencies)] = np.mean([sum(freq.values()) for freq in background_frequencies.values()])
    avgdl = dls[len(background_frequencies)]
    
    ftd = 1. * frequencies[d][t] / frequencies[d].most_common(1)[0][1]
    ld = sum(frequencies[d].values())
    
    tf = (ftd * (k1 + 1)) / (ftd + k1 * (1 - b + b * ld / avgdl))
    idf = log((N - nt + 0.5) / (nt + 0.5))
    return tf * idf


train_data = pd.DataFrame([
    {'id': doc_id + ':' + str(i), 'token': candidate} 
    for doc_id, candidates in train_candidates.items()
        for i, candidate in enumerate(candidates)
])
train_data.set_index('id', inplace=True)
print(train_data.shape)
train_data.head(100)


test_data = pd.DataFrame([
    {'id': doc_id + ':' + str(i), 'token': candidate} 
    for doc_id, candidates in test_frequencies.items()
        for i, candidate in enumerate(candidates)
])
test_data.set_index('id', inplace=True)
print(test_data.shape)
test_data.head()


train_data['tf'] = [tf(d=i.split(':')[0], t=row['token'], frequencies=train_frequencies) for i, row in tqdm(train_data.iterrows(), total=len(train_data))]
train_data['idf'] = [idf(t=row['token'], frequencies=train_frequencies) for i, row in tqdm(train_data.iterrows(), total=len(train_data))]
train_data['tf-idf'] = train_data['tf'] * train_data['idf']
train_data['bm25'] = [bm25(t=row['token'], d=i.split(':')[0], frequencies=train_frequencies, background_frequencies=train_frequencies) for i, row in tqdm(train_data.iterrows(), total=len(train_data))]
train_data['len'] = [len(row['token']) for i, row in tqdm(train_data.iterrows(), total=len(train_data))]
train_data.head()


test_data['tf'] = [tf(d=i.split(':')[0], t=row['token'], frequencies=test_frequencies) for i, row in tqdm(test_data.iterrows(), total=len(test_data))]
test_data['idf'] = [idf(t=row['token'], frequencies=test_frequencies) for i, row in tqdm(test_data.iterrows(), total=len(test_data))]
test_data['tf-idf'] = test_data['tf'] * test_data['idf']
test_data['bm25'] = [bm25(t=row['token'], d=i.split(':')[0], frequencies=test_frequencies, background_frequencies=train_frequencies) for i, row in tqdm(test_data.iterrows(), total=len(test_data))]
test_data['len'] = [len(row['token']) for i, row in tqdm(test_data.iterrows(), total=len(test_data))]
test_data.head()


with open('ake-datasets/datasets/Inspec/references/train.uncontr.json', 'r') as f:
    target = json.load(f)
    target = {doc_id: [k[0] for k in keyphrases] for doc_id, keyphrases in target.items()}
train_data['class'] = [int(row['token'] in target[i.split(':')[0]]) for i, row in tqdm(train_data.iterrows(), total=len(train_data))]
train_data.head()


with open('ake-datasets/datasets/Inspec/references/test.uncontr.json', 'r') as f:
    target = json.load(f)
    target = {doc_id: [k[0] for k in keyphrases] for doc_id, keyphrases in target.items()}
test_data['class'] = [int(row['token'] in target[i.split(':')[0]]) for i, row in tqdm(test_data.iterrows(), total=len(test_data))]
test_data.head()


# ## xgboost
import xgboost as xgb

X_train = train_data.loc[:, ~train_data.columns.isin(['class', 'id', 'token'])].values
y_train = train_data['class'].values
X_test = test_data.loc[:, ~test_data.columns.isin(['class', 'id', 'token'])].values
y_test = test_data['class'].values


D_train = xgb.DMatrix(X_train, label=y_train)
D_test = xgb.DMatrix(X_test, label=y_test)


from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)

weight = [class_weights[c] for c in y_train]


model = xgb.XGBClassifier(max_depth=5, gpu_id=0)
model.fit(X_train, y_train,
          sample_weight=weight,
          eval_set=[(X_test, y_test)],
          eval_metric='logloss', 
          verbose=True, 
          early_stopping_rounds=10)


res = model.predict(X_test, output_margin=True)
res_pred = {(i.split(':')[0], row['token']):  prob for (i, row), prob in zip(test_data.iterrows(), res)}


def score(t, d):
    """
    :param t: term
    :param d: document-id in test dataset
    """
    return res_pred[(d, t)] if (d, t) in res_pred else 0


def extract_keyphrases(doc_id, nb_keywords=5):
    scores = {candidate: score(candidate, doc_id) for candidate in test_candidates[doc_id]}
    scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)[:nb_keywords]
    return [keyphrase for keyphrase, score in scores]

predictions = {doc_id: extract_keyphrases(doc_id, nb_keywords=5) for doc_id, doc in tqdm(test_sentences.items())}
predictions = {doc_id: [sno.stem(candidate) for candidate in candidates] for doc_id, candidates in predictions.items()}
target = {doc_id: [sno.stem(candidate) for candidate in candidates] for doc_id, candidates in target.items()}

def avg_precisoin(pred, targ):
    res, nb_correct = 0, 0
    for i, p in enumerate(pred):
        if p in targ:
            nb_correct += 1
            res += nb_correct / (i + 1)
    return res / len(targ)


results = []
for doc_id in sorted(predictions.keys()):
    p = set(predictions[doc_id])
    t = set(target[doc_id])

    # We always predict 5 keywords
    precision = 0 if len(p) == 0 else len(p.intersection(t)) / len(p)
    recall = 0 if len(t) == 0 else len(p.intersection(t)) / len(t)
    results.append({
        'doc_id':      doc_id,
        'precision':   precision,
        'recall':      recall,
        'f1':          0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall),
        'precision@5': len(p.intersection(t)) / 5.,
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
print(results)

