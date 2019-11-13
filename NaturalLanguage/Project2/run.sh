# lemma2noun
fstcompile --isymbols=syms.txt --osymbols=syms.txt  lemma2noun.txt | fstarcsort > FINALtransducers/lemma2noun.fst
fstdraw    --isymbols=syms.txt --osymbols=syms.txt --portrait FINALtransducers/lemma2noun.fst | dot -Tpdf  > FINALpdf/lemma2noun.pdf

# lemma2adverb
fstcompile --isymbols=syms.txt --osymbols=syms.txt  lemma2adverb.txt | fstarcsort > FINALtransducers/lemma2adverb.fst
fstdraw    --isymbols=syms.txt --osymbols=syms.txt --portrait FINALtransducers/lemma2adverb.fst | dot -Tpdf  > FINALpdf/lemma2adverb.pdf

# lemma2verbip
fstcompile --isymbols=syms.txt --osymbols=syms.txt  lemma2verbip.txt | fstarcsort > FINALtransducers/lemma2verbip.fst
fstdraw    --isymbols=syms.txt --osymbols=syms.txt --portrait FINALtransducers/lemma2verbip.fst | dot -Tpdf  > FINALpdf/lemma2verbip.pdf

# lemma2verbis
fstcompile --isymbols=syms.txt --osymbols=syms.txt  lemma2verbis.txt | fstarcsort > FINALtransducers/lemma2verbis.fst
fstdraw    --isymbols=syms.txt --osymbols=syms.txt --portrait FINALtransducers/lemma2verbis.fst | dot -Tpdf  > FINALpdf/lemma2verbis.pdf

# lemma2verbif
fstcompile --isymbols=syms.txt --osymbols=syms.txt  lemma2verbif.txt | fstarcsort > FINALtransducers/lemma2verbif.fst
fstdraw    --isymbols=syms.txt --osymbols=syms.txt --portrait FINALtransducers/lemma2verbif.fst | dot -Tpdf  > FINALpdf/lemma2verbif.pdf


# lemma2verb
fstunion   FINALtransducers/lemma2verbip.fst FINALtransducers/lemma2verbis.fst tmp.fst
fstunion   tmp.fst FINALtransducers/lemma2verbif.fst FINALtransducers/lemma2verb.fst
rm         tmp.fst
fstdraw    --isymbols=syms.txt --osymbols=syms.txt --portrait FINALtransducers/lemma2verb.fst | dot -Tpdf  > FINALpdf/lemma2verb.pdf

# lemma2word
fstunion   FINALtransducers/lemma2verb.fst FINALtransducers/lemma2noun.fst tmp.fst
fstunion   tmp.fst FINALtransducers/lemma2adverb.fst FINALtransducers/lemma2word.fst
rm         tmp.fst
fstdraw    --isymbols=syms.txt --osymbols=syms.txt --portrait FINALtransducers/lemma2word.fst | dot -Tpdf  > FINALpdf/lemma2word.pdf


# word2lemma
fstinvert  FINALtransducers/lemma2word.fst FINALtransducers/word2lemma.fst
fstdraw    --isymbols=syms.txt --osymbols=syms.txt --portrait FINALtransducers/word2lemma.fst | dot -Tpdf  > FINALpdf/word2lemma.pdf


# FINALexamples lemma2verb.fst, lemma2word.fst, word2lemma.fst
WORDS=(cartucho jogo choro)
LEMMA_NOUNS=(choro+N+ms choro+N+mp choro+N+fs)
LEMMA_VERBS=(jogar+V+ip+1s chorar+V+ip+1s jogar+V+if+2s)

for ID in 0 1 2
do
  # WORD
  WORD=${WORDS[ID]}
  python3 word2fst.py -s syms.txt $WORD > FINALexamples/$ID.txt
  fstcompile --isymbols=syms.txt --osymbols=syms.txt  FINALexamples/$ID.txt | fstarcsort > FINALexamples/test$((ID+1))_word.fst
  fstdraw    --isymbols=syms.txt --osymbols=syms.txt --portrait FINALexamples/test$((ID+1))_word.fst | dot -Tpdf  > FINALexamples/test$((ID+1))_word.pdf
  fstcompose FINALexamples/test$((ID+1))_word.fst  FINALtransducers/word2lemma.fst FINALexamples/test$((ID+1))_word2lemma.fst
  fstdraw    --isymbols=syms.txt --osymbols=syms.txt --portrait FINALexamples/test$((ID+1))_word2lemma.fst | dot -Tpdf  > FINALexamples/test$((ID+1))_word2lemma.pdf

  # LEMMA_NOUN
  LEMMA_NOUN=${LEMMA_NOUNS[ID]}
  python3 word2fst.py -s syms.txt $LEMMA_NOUN > FINALexamples/$ID.txt
  fstcompile --isymbols=syms.txt --osymbols=syms.txt  FINALexamples/$ID.txt | fstarcsort > FINALexamples/test$((ID+1))_lemma_noun.fst
  fstdraw    --isymbols=syms.txt --osymbols=syms.txt --portrait FINALexamples/test$((ID+1))_lemma_noun.fst | dot -Tpdf  > FINALexamples/test$((ID+1))_lemma_noun.pdf
  fstcompose FINALexamples/test$((ID+1))_lemma_noun.fst  FINALtransducers/lemma2noun.fst FINALexamples/test$((ID+1))_lemma2noun.fst
  fstdraw    --isymbols=syms.txt --osymbols=syms.txt --portrait FINALexamples/test$((ID+1))_lemma2noun.fst | dot -Tpdf  > FINALexamples/test$((ID+1))_lemma2noun.pdf

  # LEMMA_VERB
  LEMMA_VERB=${LEMMA_VERBS[ID]}
  python3 word2fst.py -s syms.txt $LEMMA_VERB > FINALexamples/$ID.txt
  fstcompile --isymbols=syms.txt --osymbols=syms.txt  FINALexamples/$ID.txt | fstarcsort > FINALexamples/test$((ID+1))_lemma_verb.fst
  fstdraw    --isymbols=syms.txt --osymbols=syms.txt --portrait FINALexamples/test$((ID+1))_lemma_verb.fst | dot -Tpdf  > FINALexamples/test$((ID+1))_lemma_verb.pdf
  fstcompose FINALexamples/test$((ID+1))_lemma_verb.fst  FINALtransducers/lemma2verb.fst FINALexamples/test$((ID+1))_lemma2verb.fst
  fstdraw    --isymbols=syms.txt --osymbols=syms.txt --portrait FINALexamples/test$((ID+1))_lemma2verb.fst | dot -Tpdf  > FINALexamples/test$((ID+1))_lemma2verb.pdf

done

rm FINALexamples/*.txt
