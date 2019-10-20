import operator
from collections import Counter
from pprint import pprint
from typing import Dict, Tuple, List

from math import log
import fire
import nltk


DOCUMENT_SEPARATOR = '$$$$'


def read(file_path: str) -> Dict[str, List[Tuple[int, int]]]:
    inverted_index = {}
    with open(file_path, 'r') as f:
        all_lines = f.read()

    for i, document in enumerate(all_lines.split(DOCUMENT_SEPARATOR)):
        words = nltk.word_tokenize(document.lower())
        occurrences = Counter(words)

        for word, nb_occurrence in occurrences.items():
            if word not in inverted_index:
                inverted_index[word] = []
            inverted_index[word].append((i, nb_occurrence))

    # Sort index by frequency
    for w in inverted_index:
        inverted_index[w] = sorted(inverted_index[w], key=lambda x: x[1])
    return inverted_index


def stats(file_path: str, *terms):
    if terms is None:
        terms = []

    inverted_index = read(file_path=file_path)
    document_ids = set()
    nb_words = 0
    for idx in inverted_index.values():
        for document_id, nb_occurrence in idx:
            document_ids.add(document_id)
            nb_words += nb_occurrence

    nb_documents = len(document_ids)
    nb_unique_words = len(inverted_index)
    print(f'#Documents: {nb_documents}, #Terms: {nb_words}, #IndividualTerms: {nb_unique_words}')
    for term in terms:
        if term not in inverted_index:
            print(f'Term {term} did not appear in the documents')
            continue
        document_frequency = len(inverted_index[term])
        max_term_frequency = inverted_index[term][-1]
        min_term_frequency = inverted_index[term][0]
        inverse_document_frequency = log(nb_documents / document_frequency)
        print(f'-------- {term} ------')
        print(f'DF {term}: {document_frequency}')
        print(f'max TF: {max_term_frequency}')
        print(f'min TF: {min_term_frequency}')
        print(f'IDF: {inverse_document_frequency}')

    sim = dot_similarity(terms, inverted_index, nb_documents)
    pprint(sim)


def dot_similarity(terms: List[str], inverted_index, nb_documents):
    a = {}
    for term in terms:
        document_frequency = len(inverted_index[term])
        inverse_document_frequency = log(nb_documents / document_frequency)

        for doc_id, freq in inverted_index[term]:
            if doc_id not in a:
                a[doc_id] = {}
            if term not in a[doc_id]:
                a[doc_id][term] = 0

            tf = 0
            for doc, freq in inverted_index[term]:
                if doc == doc_id:
                    tf = freq
                    break

            a[doc_id][term] += tf * inverse_document_frequency
    return a


def main(file_path):
    inverted_index = read(file_path=file_path)
    document_ids = set()
    nb_words = 0
    for idx in inverted_index.values():
        for document_id, nb_occurrence in idx:
            document_ids.add(document_id)
            nb_words += nb_occurrence

    nb_documents = len(document_ids)
    term = input('Term: ')
    while term:
        sim = dot_similarity(terms=[term], inverted_index=inverted_index, nb_documents=nb_documents)
        most_similar = {}
        for doc_id, similarities in sim.items():
            most_similar[doc_id] = sim[doc_id][term]

        most_similar = sorted(most_similar.items(), key=operator.itemgetter(1), reverse=True)
        print(f'Term {term}: {most_similar}')

        term = input('Term: ')


if __name__ == '__main__':
    fire.Fire(main)
