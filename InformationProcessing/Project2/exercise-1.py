import itertools
import operator

import networkx as nx
import spacy
from nltk import everygrams
from tqdm import tqdm_notebook as tqdm

with open('./alice.txt', 'r') as f:
    alice = f.read().lower()

nlp = spacy.load('en_core_web_sm')
doc = nlp(alice)

G = nx.Graph(name='Alice')
for sentence in tqdm(doc.sents):
    tokens = [str(t) for t in sentence if t.is_alpha and not t.is_stop]
    grams = everygrams(tokens, min_len=1, max_len=3)
    grams = [' '.join(g) for g in grams]
    G.add_nodes_from(grams)
    G.add_edges_from(itertools.combinations(grams, 2))

rank = nx.pagerank(G, alpha=1 - 0.15, max_iter=50, weight=None)
print(sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[:5])
