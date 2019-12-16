import json
import operator
from glob import glob
from xml.etree import ElementTree

import fire
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm_notebook as tqdm

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 500)

sno = SnowballStemmer('english')


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


def extract_keyphrases(vec, feature_names, nb_keywords=5):
    feature_index = vec.nonzero()[1]
    tfidf_scores = zip(feature_index, [vec[0, x] for x in feature_index])
    # Scale scores by n-gram length
    scores = {feature_names[i]: s * len(feature_names[i].split()) for i, s in tfidf_scores}
    scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)[:nb_keywords]
    return [keyphrase for keyphrase, score in scores]


def avg_precisoin(pred, targ):
    res, nb_correct = 0, 0
    for i, p in enumerate(pred):
        if p in targ:
            nb_correct += 1
            res += nb_correct / (i + 1)
    return res / len(targ)


def evaluate(predictions, target):
    results = []
    for doc_id in sorted(predictions.keys()):
        p = set(predictions[doc_id])
        t = set(target[doc_id])

        # We always predict 5 keywords
        precision = 0 if len(p) == 0 else len(p.intersection(t)) / len(p)
        recall = 0 if len(t) == 0 else len(p.intersection(t)) / len(t)
        results.append({
            'doc_id': doc_id,
            'precision': precision,
            'recall': recall,
            'f1': 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall),
            'precision@5': len(p.intersection(t)) / 5.,
            'av_prec': avg_precisoin(p, t)
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


def main():
    train_sentences = read('ake-datasets/datasets/Inspec/train')
    test_sentences = read('ake-datasets/datasets/Inspec/test')
    print(len(train_sentences), len(test_sentences))

    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 3))
    trainvec = vectorizer.fit_transform(train_sentences.values())
    feature_names = vectorizer.get_feature_names()

    with open('ake-datasets/datasets/Inspec/references/test.uncontr.json', 'r') as f:
        target = json.load(f)
        target = {doc_name: [k[0] for k in keyphrases] for doc_name, keyphrases in target.items()}

    predictions = {}
    for doc_id, doc in test_sentences.items():
        vec = vectorizer.transform([doc])[0]
        keyphrases = extract_keyphrases(vec, feature_names=feature_names, nb_keywords=5)
        predictions[doc_id] = keyphrases

    predictions = {doc_id: [sno.stem(candidate) for candidate in candidates] for doc_id, candidates in
                   predictions.items()}
    target = {doc_id: [sno.stem(candidate) for candidate in candidates] for doc_id, candidates in target.items()}
    evaluate(predictions=predictions, target=target)


if __name__ == '__main__':
    fire.Fire(main)
