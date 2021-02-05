# Vulnerability 3: Missing user validation allows user to read any stored post

- Vulnerability: Broken access control
- Where: `id` in `GET` request of post form editation
- Impact: Allows access to any users's private/friend post

## Steps to reproduce

1. Change value of the `id` paramether in `GET /edit_post` request to any sequentional number starting from 1.

[(POC)](vuln3.py)
