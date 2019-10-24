from typing import Dict, Tuple
from xml.etree import ElementTree

import fire
import numpy as np
import pandas as pd
import time
import spacy
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm

# Please download spaCy for Portuguese before using
# `python -m spacy download pt_core_news_sm`
nlp = spacy.load('pt_core_news_sm')
sno = SnowballStemmer('portuguese')


def read_data(xml_file_path):
    doc = ElementTree.parse(xml_file_path)
    data = []
    reference: Dict[str, Tuple[int, str]] = {}

    for document in doc.findall('documento'):
        for faq in document.find('faq_list').findall('faq'):
            answer = faq.find('resposta')
            idx = answer.attrib['id']
            answer = answer.text.strip()

            for question in faq.find('perguntas').findall('pergunta'):
                question = question.text.strip()
                if question == '' or question in reference:
                    continue
                data.append((question, answer, idx))
                reference[question] = (idx, answer)

    return pd.DataFrame(data, columns=['question', 'answer', 'answer_id'])


def get_filters(q: str):
    doc = nlp(q)
    return {
        'tokenized': ' '.join([token.text.lower() for token in doc]),
        'annotated': ' '.join([token.text.lower() + ' ' + token.pos_ for token in doc]),
        'filtered': ' '.join([token.text.lower() for token in doc if token.is_alpha and not token.is_stop]),
        'filter_annotated': ' '.join([token.text.lower() + ' ' + token.pos_ for token in doc if token.is_alpha and not token.is_stop]),
        'lemma_filtered': ' '.join([token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]),
        'stem_filtered': ' '.join([sno.stem(token.text.lower()) for token in doc if token.is_alpha and not token.is_stop]),
        'stem': ' '.join([sno.stem(token.text.lower()) for token in doc]),
    }


def add_experimental_columns(data: pd.DataFrame):
    filters_applied = []

    for i, row in tqdm(data.iterrows(), total=len(data)):
        filters_applied.append(get_filters(row['question']))

    data['question'] = [item['tokenized'] for item in filters_applied]
    data['filtered'] = [item['filtered'] for item in filters_applied]
    data['annotated'] = [item['annotated'] for item in filters_applied]
    data['filter_annotated'] = [item['filter_annotated'] for item in filters_applied]
    data['lemma_filtered'] = [item['lemma_filtered'] for item in filters_applied]
    data['stem_filtered'] = [item['stem_filtered'] for item in filters_applied]
    data['stem'] = [item['stem'] for item in filters_applied]
    return data


cache = {}
vectorizer = None


def tfidf(a, b):
    if a not in cache: cache[a] = vectorizer.transform([a])
    if b not in cache: cache[b] = vectorizer.transform([b])
    avec = cache[a]
    bvec = cache[b]
    cosine_similarities = linear_kernel(avec, bvec).flatten()
    return 1 - cosine_similarities[0]


def query(q: str, corpus: pd.DataFrame, distance, threshold=10., column='question'):
    """Return answer_id with the most similar question"""
    distances = [distance(q, row[column]) for i, row in corpus.iterrows()]
    best_match = int(np.argmin([distances]))

    if distances[best_match] >= threshold:
        return 0, distances[best_match]
    return corpus.iloc[best_match]['answer_id'], distances[best_match]


def main(corpus_xml_path: str = 'KB.xml', test_path: str = 'test.txt', output_path: str = 'resultados.txt',
         column='stem'):
    start = time.time()
    data = read_data(xml_file_path=corpus_xml_path)
    data = add_experimental_columns(data)

    global cache
    global vectorizer
    cache = {}
    vectorizer = TfidfVectorizer()
    vectorizer.fit(data[column].values)

    with open(test_path, 'r') as f, open(output_path, 'w') as out:
        for q in tqdm(f):
            filters = get_filters(q)
            ans = query(filters[column],
                        corpus=data,
                        distance=tfidf,
                        threshold=0.6,
                        column=column)

            out.write(f'{ans[0]}\n')

    print('Elapsed time:', time.time() - start)


if __name__ == '__main__':
    fire.Fire(main)
