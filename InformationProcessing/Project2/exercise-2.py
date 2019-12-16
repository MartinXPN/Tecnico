import itertools
import json
import operator
from collections import Counter
from glob import glob
from xml.etree import ElementTree

import networkx as nx
import numpy as np
import pandas as pd
import spacy
from math import log
from nltk import everygrams
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
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
            sentences.append(' '.join([token.find('lemma').text.lower()
                                       for token in sentence.find('tokens').findall('token')]))

        docs[doc_path.split('/')[-1].split('.')[0]] = '\n'.join(sentences)
    return docs


train_docs = read('../Project1/ake-datasets/datasets/Inspec/train')
test_docs = read('../Project1/ake-datasets/datasets/Inspec/test')
len(train_docs), len(test_docs)

with open('../Project1/ake-datasets/datasets/Inspec/references/test.uncontr.json', 'r') as f:
    target = json.load(f)
    target = {doc_name: [k[0] for k in keyphrases] for doc_name, keyphrases in target.items()}
target['193']

vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 3))
trainvec = vectorizer.fit_transform(train_docs.values())
feature_names = vectorizer.get_feature_names()


def tf_idf(vec, feature_names):
    feature_index = vec.nonzero()[1]
    tfidf_scores = zip(feature_index, [vec[0, x] for x in feature_index])
    tfidf_scores = {feature_names[i]: s for i, s in tfidf_scores}
    return tfidf_scores


def avg_precisoin(pred, targ):
    pred = set([sno.stem(p) for p in pred])
    targ = set([sno.stem(t) for t in targ])
    res, nb_correct = 0, 0
    for i, p in enumerate(pred):
        if p in targ:
            nb_correct += 1
            res += nb_correct / (i + 1)
    return res / len(targ)


test_vecs = vectorizer.transform(test_docs.values())
test_docs = {doc_id: (test_docs[doc_id], vec) for doc_id, vec in zip(test_docs, test_vecs)}


def score(doc_id, doc, get_weight, get_personalization=None, weight='weight'):
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
    top = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[:5]
    top_keywords = [k for k, s in top]
    return avg_precisoin(pred=top_keywords, targ=target[doc_id])


# ## Original
precision = [score(doc_id, doc, get_weight=lambda x, y: 1, weight=None)
             for doc_id, (doc, vec) in tqdm(test_docs.items())]
print(np.mean(precision))

# ## TF-IDF scores as personalization
precision = []
for doc_id, (doc, vec) in tqdm(test_docs.items()):
    tfidf_scores = tf_idf(vec, feature_names)

    s = score(doc_id, doc, get_weight=lambda x, y: 1,
              get_personalization=lambda x: tfidf_scores[x] if x in tfidf_scores else 0)
    precision.append(s)

print(np.mean(precision))

# ## Position score as personalization
precision = []
for doc_id, (doc, vec) in tqdm(test_docs.items()):
    d = nlp(doc)
    d = ' '.join([str(t) for t in d if t.is_alpha and not t.is_stop])
    s = score(doc_id, doc, get_weight=lambda x, y: 1,
              get_personalization=lambda x: 1 / (1 + log(1 + d.find(x))))
    precision.append(s)

print(np.mean(precision))

# ## Co-occurance weighting
W = Counter()
for doc_id, doc in tqdm(train_docs.items()):
    doc = nlp(doc)
    for sentence in doc.sents:
        tokens = [str(t) for t in sentence if t.is_alpha and not t.is_stop]
        grams = everygrams(tokens, min_len=1, max_len=3)
        grams = [' '.join(g) for g in grams]
        for w1, w2 in itertools.combinations(grams, 2):
            W[(w1, w2)] += 1
            W[(w2, w1)] += 1

precision = [score(doc_id, doc, get_weight=lambda x, y: W[(x, y)])
             for doc_id, (doc, vec) in tqdm(test_docs.items())]
print(np.mean(precision))

# ## Word vector similarity
from sklearn.metrics.pairwise import cosine_similarity
from fasttext import load_model

model = load_model('./cc.en.300.bin')


def weight(w1, w2):
    v1 = model.get_sentence_vector(w1)
    v2 = model.get_sentence_vector(w2)
    return cosine_similarity([v1], [v2])[0]


precision = [score(doc_id, doc, get_weight=weight)
             for doc_id, (doc, vec) in tqdm(test_docs.items())]
print(np.mean(precision))

# ## Ensemble
precision = []

for doc_id, (doc, vec) in tqdm(test_docs.items()):
    d = nlp(doc)
    d = ' '.join([str(t) for t in d if t.is_alpha and not t.is_stop])
    s = score(doc_id, doc, get_weight=weight,
              get_personalization=lambda x: 1 / (1 + log(1 + d.find(x))))
    precision.append(s)

print(np.mean(precision))
