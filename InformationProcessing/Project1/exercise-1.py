import operator
import fire
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keyphrases(vec, feature_names, nb_keywords=5):
    feature_index = vec.nonzero()[1]
    tfidf_scores = zip(feature_index, [vec[0, x] for x in feature_index])
    # Scale scores by n-gram length
    scores = {feature_names[i]: s * len(feature_names[i].split()) for i, s in tfidf_scores}
    scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)[:nb_keywords]
    return [keyphrase for keyphrase, score in scores]


def main(doc_path='alice.txt'):
    train = fetch_20newsgroups(subset='train')
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', token_pattern=u'(?u)[a-zA-Z]+',
                                 ngram_range=(1, 3))
    trainvec = vectorizer.fit_transform(train.data)

    with open(doc_path, 'r') as f:
        alice = f.read()
        docs = [alice]  # alice.split('\n\n')
    docs = [doc.replace('\n', ' ').lower().strip() for doc in docs]

    testvec = vectorizer.transform(docs)
    feature_names = vectorizer.get_feature_names()

    for doc, vec in zip(docs, testvec):
        keyphrases = extract_keyphrases(vec, feature_names=feature_names, nb_keywords=5)
        print(', '.join(keyphrases))
        # f.write(', '.join(keyphrases) + '\n')


if __name__ == '__main__':
    fire.Fire(main)
