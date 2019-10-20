#!/bin/bash

# Convert to lowercase
tr '[:upper:]' '[:lower:]' < alice.txt > alice.lower
tr -sc 'A-Za-z' '\n' < alice.lower > texto.unigrams

# Remove the first line, then the first and the second, ...
tail +2 texto.unigrams > tmp1 # sintaxe alternativa -n +2
tail +2 tmp1 > tmp2
tail +2 tmp2 > tmp3

# Align files in order to obtain bigrams, trigrams,...
echo "Cria bigrams, trigrams and quadrigrams"
paste -d " " texto.unigrams tmp1 | less > texto.bigrams
paste -d " " texto.unigrams tmp1 tmp2 | less > texto.trigrams
paste -d " " texto.unigrams tmp1 tmp2 tmp3 | less > texto.quadrigrams

# Show the 10 most frequent n-grams
echo "Bigrams mais frequentes"
sort texto.bigrams | uniq -c | sort -nr | head -n10

echo "Trigrams mais frequentes"
sort texto.trigrams | uniq -c | sort -nr | head -n10

echo "Quadrigrams mais frequentes"
sort texto.quadrigrams | uniq -c | sort -nr | head -n10

# Build the final files
sort texto.unigrams | uniq -c | sort -nr > contagensUnigramas.txt
sort texto.bigrams | uniq -c | sort -nr > contagensBigramas.txt
sort texto.trigrams | uniq -c | sort -nr > contagensTrigramas.txt
sort texto.quadrigrams | uniq -c | sort -nr > contagensQuadrigramas.txt