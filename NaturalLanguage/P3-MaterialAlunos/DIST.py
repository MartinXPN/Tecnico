# -*- coding: utf-8 -*-
import os, re, codecs
import nltk
from nltk.metrics.scores import accuracy
from nltk import sent_tokenize, word_tokenize

from nltk.metrics.distance import edit_distance
from nltk.metrics.distance import jaccard_distance


#-----------------------------------------
# Use this with the latest version of nltk and Python 3 
# In case of dispair (maybe) you can try to remove all the accents with: 
# iconv -f UTF-8 -t ascii//TRANSLIT Corpora/dist-desen.txt | tr -d "'~^"


# Input
# dist-desen, dist-teste, dist-treino (ou a versão sem acentos dos primeiros)
# Each line has the form:
# TAG SENTENCE
# 
# Target: find the correct tags to the test set
#
# Please:
# 1) Use the development set for development
# 2) Use the training set as a knowledge base
# 3) Use the test only at the end, in the final evaluation
# 4) Baseline = edit-distance
# 5) Evaluation measure = accuracy

# What should you do:
# (0) Run the baseline
# (1) Try to improve results with extra pre-processing (lowercasing, stopwords, stemming,...)
# (2) Test other similarity measures.  You can even invent you own measure. 
# (3) Test with the test file.
#-----------------------------------------

#--------------
# Stopword to remove. Add more stopwords...
stopWords = ['a', 'o', 'teu']

#--------------
# Returns the list of tags (results) and the sentences that were considered to be the closest
# to other measures checks:
# http://www.nltk.org/_modules/nltk/metrics/distance.html
# Notice that there are small differences between measures. For instance, Jaccard receives sets and not lists as input:
# (call: jaccard_distance(set(listaFrasesTreino[j]), set(listaFrasesDesenvolvimento[i]))
# Jaccard is 0 if =
# For some measures it is possible that you need to touch the initial value of best
# edit-distance:
#edit_distance(listaFrasesTreino[j], listaFrasesDesenvolvimento[i]), best = 10000, result < best

def mainFunction(listaTagsTreino, listaFrasesTreino, listaFrasesDesenvolvimento):
	results = []
	bestSentences = []
	i = 0
	while i < len(listaFrasesDesenvolvimento):
		j = 0
		best = 1000
		tagId = "VOID"
		bestSentence = ""
		while j < len(listaFrasesTreino):
			# It is really a distance and not a similarity measure (1-similarity)
			#result = jaccard_distance(set(listaFrasesTreino[j].split()), set(listaFrasesDesenvolvimento[i].split()))
			result = edit_distance(listaFrasesTreino[j].split(), listaFrasesDesenvolvimento[i].split())
			#print(result)
			if result < best:
				tagId = listaTagsTreino[j]
				bestSentence = listaFrasesTreino[j]
				best = result
			j = j + 1
		results.append(tagId)
		bestSentences.append(bestSentence)
		i = i + 1
	return results, bestSentences

def main():

	# Splits desen and train in 2: tags and sentences
	
	# Process desen
	listaTagsDesenvolvimento = extrai('Corpora/dist-desen.txt',1)
	listaFrasesDesenvolvimento = extrai('Corpora/dist-desen.txt', 2)

	# Process treino
	listaTagsTreino = extrai('Corpora/dist-treino.txt', 1)
	listaFrasesTreino = extrai('Corpora/dist-treino.txt', 2)

	#----- Pre-processing-----
#	listaFrasesDesenvolvimento = preProc(listaFrasesDesenvolvimento)
#	listaFrasesTreino = preProc(listaFrasesTreino)

	#----- Remove stopWords-----
#	listaFrasesDesenvolvimento = removeStopWords(listaFrasesDesenvolvimento, stopWords)
#	listaFrasesTreino = removeStopWords(listaFrasesTreino, stopWords)

	#----- Stemming -----
#	listaFrasesDesenvolvimento = tokStem(listaFrasesDesenvolvimento)
#	listaFrasesTreino = tokStem(listaFrasesTreino)

	# Call the main function
	listaTagsEstimada = mainFunction(listaTagsTreino , listaFrasesTreino, listaFrasesDesenvolvimento)[0]
	fraseMaisProxima = mainFunction(listaTagsTreino , listaFrasesTreino, listaFrasesDesenvolvimento)[1]

	# Show results
	for a, b, c, d in zip(listaFrasesDesenvolvimento, listaTagsEstimada, listaTagsDesenvolvimento, fraseMaisProxima):
		print("Sentence to evaluate: ", a)
		print("Suggested Tag: ", b)
		print("Correct Tag: ", c)
		print("Closest sentence: ", d, "\n\n")

	# Find accuracy
	print ("Accuracy:", accuracy(listaTagsDesenvolvimento, listaTagsEstimada))

#--------------------------------------------------------
# Aux
#--------------------------------------------------------
def preProc(Lista):
	perguntas = []
	for l in Lista:
		# ELIMINA ACENTOS
		l = re.sub(u"ã", 'a', l)
		l = re.sub(u"á", "a", l)
		l = re.sub(u"à", "a", l)
		l = re.sub(u"õ", "o", l)
		l = re.sub(u"ô", "o", l)
		l = re.sub(u"ó", "o", l)
		l = re.sub(u"é", "e", l)
		l = re.sub(u"ê", "e", l)
		l = re.sub(u"í", "i", l)
		l = re.sub(u"ú", "u", l)
		l = re.sub(u"ç", "c", l)
		l = re.sub(u"Ã", 'A', l)
		l = re.sub(u"Á", "A", l)
		l = re.sub(u"À", "A", l)
		l = re.sub(u"Õ", "O", l)
		l = re.sub(u"Ô", "O", l)
		l = re.sub(u"Ô", "O", l)
		l = re.sub(u"Ó", 'O', l)
		l = re.sub(u"Í", "I", l)
		l = re.sub(u"Ú", "U", l)
		l = re.sub(u"Ç", "C", l)
		l = re.sub(u"É", "E", l)
		# TUDO EM MINÚSCULAS
		l = l.lower()
		# ELIMINA PONTUAÇÃO
		l = re.sub("[?|\.|!|:|,|;]", '', l)
		# fica so com as perguntas
		l = re.sub("^\w+\t+[^\w]", '', l)
		perguntas.append(str(l))
	return perguntas

#------------------------------
# Remove stopwords
#------------------------------

# It is case insensitive
def removeStopWords(list, stopWordList):
	perguntas = []
	for sentence in list:
		sentence = sentence.split()
		frase = []
		for word in sentence:
			if word.lower() not in stopWordList:
				frase.append(word)
			fraseAux = ' '.join(frase)	
		perguntas.append(fraseAux)
	return perguntas

#------------------------------
# Tokenization and stemmer
#------------------------------

def tokStem(perguntas):
	perguntas_tok_stem = []
	stemmer = nltk.stem.RSLPStemmer()
	for l in perguntas:
		l = nltk.word_tokenize(l)
		l1 = []
		for word in l:
			word = stemmer.stem(word)
			l1.append(word)
		l = ' '.join(l1)
		perguntas_tok_stem.append(l)
	return perguntas_tok_stem

#---------------
# Print a list
#---------------
def print_list(list):
	j = 0
	while j < len(list):
		print (list[j])
		j = j + 1

#---------------
# Print both lists, side by side
# They should have the same size
#---------------
def print_lists(list1, list2):
	j = 0
	while j < len(list1):
		print (j, list1[j] + "\t" + list2[j])
		j = j + 1

#---------------
# Input (each line):
# TAG SENTENCE
# numColuna = 1 => TAG
# numColuna = 2 => SENTENCE 
# Return:
# Tags vector if numColuna = 1
# Sentences vector if numColuna = 2
#----------------
def extrai(nameFile, numColuna):
	file = open(nameFile, 'rU')
	tags = []
	sentences = []
	for line in file:
		field = re.search(r"(\w+[^\s])\t+(.+)", line)
		if field is None:
			print ("nada")
		else:
			tag = field.group(1)
			sentence = field.group(2)
			tags.append(tag)
			sentences.append(sentence)
	file.close()
	if numColuna == 1:
		return tags
	if numColuna == 2:
		return sentences

main()

