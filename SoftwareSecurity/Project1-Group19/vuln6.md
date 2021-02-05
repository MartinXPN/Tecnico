# Vulnerability 6: SQL Injection in edit post form allows to change any/all post

- Vulnerability: SQL Injection
- Where: `id`, `content` and `type` in `POST /edit_post`
- Impact: Allows to change the database data

## Steps to reproduce user password exstraction

1. Register a new user (if you already have an user you can go to step 2)
2. Login as that user
3. Send `POST` request to `/edit_post` with parameters `id={{payload}}content={{new_post}}&type=Private`
4. Insert payload:
    1. `11' or author like ' + {{target_username}}`
    2. Causes change of all posts of the `target_username` user to new defined `content`

[(POC)](vuln6.py)