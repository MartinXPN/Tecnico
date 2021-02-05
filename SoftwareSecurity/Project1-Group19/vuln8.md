# Vulnerability 8: CSRF in create post

- Vulnerability: CSRF
- Where: Form `POST /create_post`
- Impact: CSRF in `create_post` allows creating a post on behalf of another user by visiting the malicious site while having an active session in the same browser. Similar exploitation is possible for other `POST` requests.

## Steps to reproduce CSRF attack

1. Register a new user (if you already have an user you can go to step 2)
2. Login as that user
3. In new browser tab open the malicious website PoC `https://qwsk.000webhostapp.com/CSRF.html`
4. Submit hidden form

[(POC)](vuln8.py)