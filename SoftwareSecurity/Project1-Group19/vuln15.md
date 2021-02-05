# Vulnerability 15: Passwords stored in plain text

- Vulnerability: Unprotected Storage of Credentials
- Where: MySQL database
- Impact: Exploiting other vulnerability attackers can read stored database
data. Storing credentials in the clear text allows a malicious actors to impersonate
other users and test their credentials on other platforms. 

## Steps to reproduce
1. Register and login a user
2. Send `GET /friends` request with paramether:
```
qwer' union select username,1,password,1,1 from Users --  
```
3. Read plain text passwords

[(POC)](vuln15.py)
