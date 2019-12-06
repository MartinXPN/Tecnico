import itertools
import json
import operator
import re
from collections import Counter
from glob import glob
from xml.etree import ElementTree

import networkx as nx
import numpy as np
import pandas as pd
import spacy
from math import log
from nltk import everygrams
from nltk import ngrams
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm_notebook as tqdm

sno = SnowballStemmer('english')
nlp = spacy.load('en_core_web_sm')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 500)


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


train_docs = read('../Project1/ake-datasets/datasets/Inspec/train')
test_docs = read('../Project1/ake-datasets/datasets/Inspec/test')
len(train_docs), len(test_docs)

pattern = re.compile(r'(((\w+~JJ)* (\w+~NN)+ (\w+~IN))?(\w+~JJ)+ (\w+~NN)+)+')

train_candidates = {doc_id: [candidate[0] for candidate in re.findall(pattern, doc)] for doc_id, doc in
                    train_docs.items()}
train_candidates = {doc_id: [' '.join([w.split('~')[0] for w in candidate.split()]) for candidate in candidates] for
                    doc_id, candidates in train_candidates.items()}
train_docs = {doc_id: ' '.join([w.split('~')[0] for w in sentences.split()]) for doc_id, sentences in
              train_docs.items()}
train_frequencies = {doc_id: Counter(
    [' '.join(gram) for gram in ngrams(doc.split(), 1)] + \
    [' '.join(gram) for gram in ngrams(doc.split(), 2)] + \
    [' '.join(gram) for gram in ngrams(doc.split(), 3)])
    for doc_id, doc in train_docs.items()}

test_candidates = {doc_id: [candidate[0] for candidate in re.findall(pattern, doc)] for doc_id, doc in
                   test_docs.items()}
test_candidates = {doc_id: [' '.join([w.split('~')[0] for w in candidate.split()]) for candidate in candidates] for
                   doc_id, candidates in test_candidates.items()}
test_docs = {doc_id: ' '.join([w.split('~')[0] for w in sentences.split()]) for doc_id, sentences in test_docs.items()}
test_frequencies = {doc_id: Counter(
    [' '.join(gram) for gram in ngrams(doc.split(), 1)] + \
    [' '.join(gram) for gram in ngrams(doc.split(), 2)] + \
    [' '.join(gram) for gram in ngrams(doc.split(), 3)])
    for doc_id, doc in test_docs.items()}
with open('../Project1/ake-datasets/datasets/Inspec/references/test.uncontr.json', 'r') as f:
    target = json.load(f)
    target = {doc_name: [k[0] for k in keyphrases] for doc_name, keyphrases in target.items()}
target['193']


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


tf(t='datum', d='1390', frequencies=train_frequencies), idf(t='datum', frequencies=train_frequencies), bm25(t='datum',
                                                                                                            d='1390',
                                                                                                            frequencies=train_frequencies,
                                                                                                            background_frequencies=train_frequencies)

train_data = pd.DataFrame([
    {'id': doc_id + ':' + str(i), 'token': candidate}
    for doc_id, candidates in train_candidates.items()
    for i, candidate in enumerate(candidates)
])
train_data.set_index('id', inplace=True)

test_data = pd.DataFrame([
    {'id': doc_id + ':' + str(i), 'token': candidate}
    for doc_id, candidates in test_frequencies.items()
    for i, candidate in enumerate(candidates)
])
test_data.set_index('id', inplace=True)

print(test_data.shape)
print(train_data.shape)
train_data.head(3)

train_data['tf'] = [tf(d=i.split(':')[0], t=row['token'], frequencies=train_frequencies) for i, row in
                    tqdm(train_data.iterrows(), total=len(train_data))]
train_data['idf'] = [idf(t=row['token'], frequencies=train_frequencies) for i, row in
                     tqdm(train_data.iterrows(), total=len(train_data))]
train_data['tf-idf'] = train_data['tf'] * train_data['idf']
train_data['bm25'] = [
    bm25(t=row['token'], d=i.split(':')[0], frequencies=train_frequencies, background_frequencies=train_frequencies) for
    i, row in tqdm(train_data.iterrows(), total=len(train_data))]
train_data['len'] = [len(row['token']) for i, row in tqdm(train_data.iterrows(), total=len(train_data))]
train_data.head()

test_data['tf'] = [tf(d=i.split(':')[0], t=row['token'], frequencies=test_frequencies) for i, row in
                   tqdm(test_data.iterrows(), total=len(test_data))]
test_data['idf'] = [idf(t=row['token'], frequencies=test_frequencies) for i, row in
                    tqdm(test_data.iterrows(), total=len(test_data))]
test_data['tf-idf'] = test_data['tf'] * test_data['idf']
test_data['bm25'] = [
    bm25(t=row['token'], d=i.split(':')[0], frequencies=test_frequencies, background_frequencies=train_frequencies) for
    i, row in tqdm(test_data.iterrows(), total=len(test_data))]
test_data['len'] = [len(row['token']) for i, row in tqdm(test_data.iterrows(), total=len(test_data))]
test_data.head()

# ## Add graph centrality score for each candidate
from sklearn.metrics.pairwise import cosine_similarity
from fasttext import load_model

model = load_model('./cc.en.300.bin')


def get_ranks(doc_id, doc, get_weight, get_personalization=None, weight='weight'):
    G = nx.Graph(name=doc_id)
    doc = nlp(doc)

    for sentence in doc.sents:
        tokens = [str(t) for t in sentence if t.is_alpha and not t.is_stop]
        grams = everygrams(tokens, min_len=1, max_len=3)
        grams = [' '.join(g) for g in grams]
        G.add_nodes_from(grams)

        edges = list(itertools.combinations(grams, 2))
        weighted_edges = [(v1, v2, get_weight(v1, v2)) for v1, v2 in edges]
        G.add_weighted_edges_from(weighted_edges)

    personalization = {node: get_personalization(node) for node in G.nodes} if get_personalization else None
    rank = nx.pagerank(G, alpha=1 - 0.15, max_iter=50, weight=weight, personalization=personalization)
    return rank


def weight(w1, w2):
    v1 = model.get_sentence_vector(w1)
    v2 = model.get_sentence_vector(w2)
    return cosine_similarity([v1], [v2])[0]


cur_doc_id = -1
centralities = []
doc = None
for i, row in tqdm(test_data.iterrows(), total=len(test_data)):
    doc_id = i.split(':')[0]
    if cur_doc_id != doc_id:
        cur_doc_id = doc_id
        doc = test_docs[doc_id]
        d = nlp(doc)
        d = ' '.join([str(t) for t in d if t.is_alpha and not t.is_stop])
        doc_ranks = get_ranks(doc_id, doc, get_weight=weight,
                              get_personalization=lambda x: 1 / (1 + log(1 + d.find(x))))

    token = row['token']
    if token not in doc_ranks:  # To remove stop words and non alphanumeric tokens later on
        centralities.append(None)
    else:
        try:
            centralities.append(doc_ranks[token][0])
        except:
            centralities.append(doc_ranks[token])

test_data['centrality'] = centralities
test_data.head()

test_data.dropna(inplace=True)
test_data.head()


def avg_precisoin(pred, targ):
    pred = set([sno.stem(p) for p in pred])
    targ = set([sno.stem(t) for t in targ])
    res, nb_correct = 0, 0
    for i, p in enumerate(pred):
        if p in targ:
            nb_correct += 1
        res += nb_correct / (i + 1)
    return 1. / len(targ) * res


test_data['doc_id'] = [i.split(':')[0] for i in test_data.index]

pd.options.mode.chained_assignment = None  # default='warn'
test_doc_groups = test_data.copy()
test_doc_groups.reset_index(inplace=True)
del test_doc_groups['id']
test_doc_groups = test_doc_groups.groupby('doc_id')


def RRFScore(ranks):
    return sum(1. / (50 + r) for r in ranks)


def rank(arr):
    order = arr.argsort()
    ranks = order.argsort()
    return ranks


def score(doc_id, df):
    ranking = {}
    for i, row in df.iterrows():
        ranking[row['token']] = RRFScore([row['tf'], row['idf'], row['tf-idf'], row['bm25'],
                                          row['len'], row['centrality']])

    top = sorted(ranking.items(), key=operator.itemgetter(1), reverse=True)[:5]
    top_keywords = [k for k, s in top]
    #     print(doc_id)
    #     print(top_keywords)
    #     print(target[doc_id])
    return avg_precisoin(pred=top_keywords, targ=target[doc_id])


precision = []
for name, group in tqdm(test_doc_groups):
    group['tf'] = rank(-group['tf'].values)
    group['idf'] = rank(-group['idf'].values)
    group['tf-idf'] = rank(-group['tf-idf'].values)
    group['bm25'] = rank(-group['bm25'].values)
    group['len'] = rank(-group['len'].values)
    group['centrality'] = rank(-group['centrality'].values)

    s = score(name, group)
    precision.append(s)
#     print(name)
#     print(group)
#     break

print(np.mean(precision))
