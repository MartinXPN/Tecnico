import operator
import random
import re


# Takes as input the name of a file in which each line has the form:
# NUM n-gram
def extrai(file_path):
    freqs = []  # keep frequencies
    ngrams = []  # keep n-grams
    with open(file_path, 'r') as f:
        for line in f:
            field = re.search(r"(\d+)\s+(.+)", line)
            if field is None:
                print("Asneira com ", line)
            else:
                freq = field.group(1)
                ngram = field.group(2)
                freqs.append(freq)
                ngrams.append(ngram)
    return freqs, ngrams


# search for the frequency of a n-gram
def searchFreq(ngrams, freqs, seq):
    if seq in ngrams:
        return int(freqs[ngrams.index(seq)])
    else:
        return 0


# Find: prob(ab) = count(ab)/count(a) -> (count(ab) +1) / (count(a) + |v|)
def prob(ab):
    [a, b] = ab.rsplit(' ', 1)
    nb_words = ab.count(' ')

    freq_a = searchFreq(grams[nb_words][0], grams[nb_words][1], a)
    freq_ab = searchFreq(grams[nb_words + 1][0], grams[nb_words + 1][1], ab)
    # return freq_ab / freq_a
    res = (freq_ab + 1) / (freq_a + len(unigrams))
    # print(f'P({ab} | {a}, {b}) = {res}')
    return res


# Calculates n-grams
uniFreq, unigrams = extrai('contagensUnigramas.txt')
biFreq, bigrams = extrai('contagensBigramas.txt')
triFreq, trigrams = extrai('contagensTrigramas.txt')
quadFreq, quadgrams = extrai('contagensQuadrigramas.txt')
grams = {
    1: [unigrams, uniFreq],
    2: [bigrams, biFreq],
    3: [trigrams, triFreq],
    4: [quadgrams, quadFreq],
}

# ------> Finds the probability of a sentence
sentence = "alice said to the king"

words = sentence.split(' ')
probability = 1
for index in range(len(words) - 1):
    probability *= prob(words[index] + ' ' + words[index + 1])

print("Probability of --", sentence, "-- is", probability)


# ------> Generates sentences of a pre defined size from a trigger word (see below)
# Check possible next words
def top_next_options(word, max_options=10):
    result = [bigram for bigram in bigrams if bigram.startswith(word)] + \
             [trigram for trigram in trigrams if trigram.startswith(word)] + \
             [quadgram for quadgram in quadgrams if quadgram.startswith(word)]
    phrase2prob = {seq: prob(seq) for seq in result}
    phrase2prob = sorted(phrase2prob.items(), key=operator.itemgetter(1), reverse=True)

    top_options = [w.split()[-1] for w, p in phrase2prob]
    # print('TOP results:', top_options[:max_options])
    return top_options[:max_options]


trigger = ["alice"]
size = 10

ngrams = 2
phrase = ' '.join(trigger)
for index in range(size):
    next_options = top_next_options(' '.join(trigger), max_options=5)
    if len(next_options) == 0:
        break

    next_word = random.choice(next_options)
    phrase = phrase + ' ' + next_word
    trigger = phrase.split(' ')
    trigger = trigger[len(trigger) - ngrams:]
    print("Sentence being generated:", phrase)
    # print('Trigger:', trigger)
