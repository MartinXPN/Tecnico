It is included a program written in python that can be useful for the elaboration of the work.


word2fst.py - given a word, generates a transducer corresponding to that word. Examples:


./word2fst.py pedro
    0 1 p p
    1 2 e e
    2 3 d d
    3 4 r r
    4 5 o o
    5

./word2fst.py -s syms.txt barco+N+ms
    0 1 b b
    1 2 a a
    2 3 r r
    3 4 c c
    4 5 o o
    5 6 +N +N
    6 7 +ms +ms
    7

