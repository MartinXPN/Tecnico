# Nouns
fstcompile --isymbols=syms.txt --osymbols=syms.txt  lemma2noun.txt | fstarcsort > lemma2noun.fst
fstdraw    --isymbols=syms.txt --osymbols=syms.txt --portrait lemma2noun.fst | dot -Tpdf  > lemma2noun.pdf

# barco+N+ms
python3 word2fst.py -s syms.txt barco+N+ms > barco.txt
fstcompile --isymbols=syms.txt --osymbols=syms.txt  barco.txt | fstarcsort > barco.fst
fstdraw    --isymbols=syms.txt --osymbols=syms.txt --portrait barco.fst | dot -Tpdf  > barco.pdf

fstcompose barco.fst lemma2noun.fst > barco2noun.fst # DOES NOT WORK
fstdraw    --isymbols=syms.txt --osymbols=syms.txt --portrait barco2noun.fst | dot -Tpdf  > barco2noun.pdf
