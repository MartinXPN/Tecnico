# Vulnerability 4: Missing user validation allows user to change any stored post

- Vulnerability: Broken access control
- Where: Change of `id` in `POST` request of post's form
- Impact: Allows making change to any users's post

## Steps to reproduce

1. Change value of the `id` paramether in `POST /edit_post` request to number of desired request to change.

[(POC)](vuln4.py)
