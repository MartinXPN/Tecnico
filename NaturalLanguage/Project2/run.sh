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
fstunion   FINALtransducers/lemma2verb.fst FINALtransducers/lemma2noun.fst FINALtransducers/lemma2word.fst
fstdraw    --isymbols=syms.txt --osymbols=syms.txt --portrait FINALtransducers/lemma2word.fst | dot -Tpdf  > FINALpdf/lemma2word.pdf


# word2lemma
fstinvert  FINALtransducers/lemma2word.fst FINALtransducers/word2lemma.fst
fstdraw    --isymbols=syms.txt --osymbols=syms.txt --portrait FINALtransducers/word2lemma.fst | dot -Tpdf  > FINALpdf/word2lemma.pdf
