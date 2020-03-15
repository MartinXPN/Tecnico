from collections import Counter
from pprint import pprint

s = """
k@Stjun”adu, 6”i~da, s”obr@ 6 ive~tw6lid”ad@ d@ s@ 6~t@sip”ar6~j~ 6S f”Erj6S d6 “paSkw6, d@ f"Orm6 6 k@ uS 6l"unuS pud"Es6~j~ fik"ar 6~j~ k"az6, n"um6 @Sp"Esi@ d@ kw6re~t"en6 disimul"ad6, 6~t"Onju K"OSt6 R@m@t"ew 6 d@siz"6~w~ p"6r6 u R@zult"adu d6 m"eZm6 Rjunj"6~w~, d@ am6J"6~.
""".strip()

print(s)

all = """
301	i	vi	v"i
302	e	vê	v"e
303	E	pé	p"E
304	a	pá	p"a
324	6	para	p6r6
322	@	de	d@
306	O	pó	p"O
307	o	avô	6v"o
308	u	tu	t"u
i~	sim	s"i~
e~	pente	p"e~t@
6~	branco	br"6~ku
o~	ponte	p"o~t@
u~	atum	6t"u~
394	j	pai	p"aj
321	w	pau	p"aw
j~	põe	p"o~j~
w~	mão	m"6~w~
101	p	pá	p"a
103	t	tu	t"u
109	k	casa	k"az6
102	b	bem	b"6~j~
104	d	dou	d"o
110	g	gato	g"atu
128	f	fé	f"E
132	s	sol	s"Ol~
134	S	chave	S"av@
129	v	vê	v"e
133	z	casa	k"az6
135	Z	já	Z"a
155	l	lá	l"a
209	l~	mal	m"al~
157	L	valha	v"aL6
124	r	caro	k"aru
123	R	carro	k"aRu
114	m	muito	m"u~j~tu
116	n	não	n"6~w~
118	J	senha	s"6J6
""".strip()

sampa = []
for line in all.split('\n'):
    if line.split('\t')[0].isdigit():
        c = line.split('\t')[1]
    else:
        c = line.split('\t')[0]
    sampa.append(c)
sampa += ['“', '”', ' ', ',', '.', '"', 'K']

sampa = sorted(sampa, key=len, reverse=True)
print(sampa)

seen_symbols = []
i = 0
while i < len(s):
    for sym in sampa:
        if s[i:].startswith(sym):
            seen_symbols.append(sym)
            i += len(sym)
            break

res = Counter({sym: 0 for sym in sampa})
res.update(Counter(seen_symbols))
pprint(res)
