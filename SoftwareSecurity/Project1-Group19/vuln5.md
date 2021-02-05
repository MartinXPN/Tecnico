# Vulnerability 5: SQL Injection in create post form allows to retrieve the database data

- Vulnerability: SQL Injection
- Where: `content` and `type` in `POST /create_post`
- Impact: Allows to retrieve the database data

## Steps to reproduce user password exstraction

1. Register a new user (if you already have an user you can go to step 2)
2. Login as that user
3. Send `POST` request to `/create_post` with parameters `content={{payload}}&type=Private`
4. Repeatedly insert payload:
    1. `aaa',(select password from Users where substr(password, {{counter}}+1), 1) = '{{char[counter]}} and username like '{{target_username}}')) -- `
    2. chars beiing list of potentional password characters.
    3. In case of redirection, character guess was not succesful.
    4. In case of database returning SQL truncatenaition error remember password character and increase counter.
    5. Changing target_username to user that we desire to extract password for.
5. (Creates a lot of private posts during the password guessing process)

[(POC)](vuln5.py)