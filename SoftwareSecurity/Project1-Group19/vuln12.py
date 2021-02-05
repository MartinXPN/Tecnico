import requests, sys, re, time, random, string

passwd = 'pentest1'
params = {
    'username': 'pentest1',
    'password': passwd 
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

resource = '/update_profile'
path = url + resource
big_pic = 'big.jpg'

params = {
    'name': (None, ''),
    'currentpassword': (None, passwd),
    'newpassword': (None, ''),
    'about': (None, ''),
    'photo': (big_pic, open('src/' + big_pic, 'rb'))
}
r = s.post(url=path, files=params)
pic = re.findall(r'photos/(.+?)big.jpg', r.text)[0]
pic = pic + big_pic

tux_pic = 'tux.png'

params = {
    'name': (None, ''),
    'currentpassword': (None, passwd),
    'newpassword': (None, ''),
    'about': (None, ''),
    'photo': (tux_pic, open('src/' + tux_pic, 'rb'))
}
r = s.post(url=path, files=params)
pic2 = re.findall(r'photos/(.+?)tux.png', r.text)[0]
pic2 = pic2 + tux_pic

resource = '/static/photos/' + pic
path = url + resource
r = s.get(url=path)
print(pic)
assert r.status_code == 200

resource = '/static/photos/' + pic2
path = url + resource
r = s.get(url=path)
print(pic2)
assert r.status_code == 200
