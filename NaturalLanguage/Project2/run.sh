# Nouns
fstcompile --isymbols=syms.txt --osymbols=syms.txt  lemma2noun.txt | fstarcsort > lemma2noun.fst
fstdraw    --isymbols=syms.txt --osymbols=syms.txt --portrait lemma2noun.fst | dot -Tpdf  > lemma2noun.pdf

# Adverbs
fstcompile --isymbols=syms.txt --osymbols=syms.txt  lemma2adverb.txt | fstarcsort > lemma2adverb.fst
fstdraw    --isymbols=syms.txt --osymbols=syms.txt --portrait lemma2adverb.fst | dot -Tpdf  > lemma2adverb.pdf

# Verbip
fstcompile --isymbols=syms.txt --osymbols=syms.txt  lemma2verbip.txt | fstarcsort > lemma2verbip.fst
fstdraw    --isymbols=syms.txt --osymbols=syms.txt --portrait lemma2verbip.fst | dot -Tpdf  > lemma2verbip.pdf

# Verbis
fstcompile --isymbols=syms.txt --osymbols=syms.txt  lemma2verbis.txt | fstarcsort > lemma2verbis.fst
fstdraw    --isymbols=syms.txt --osymbols=syms.txt --portrait lemma2verbis.fst | dot -Tpdf  > lemma2verbis.pdf
