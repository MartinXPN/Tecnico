# Vulnerability 14: Missing HTTPS encryption

- Vulnerability: Missing Encryption of Sensitive Data
- Where: `/login` paramethers `username` and `password`
- Impact: User credential are send over insecure channel in clear text

## Steps to reproduce
1. Observe network traffic (i.e. using network proxy).
2. Fill and send login form.
3. Read cleartext credential from request body.

[(POC)](vuln14.py)
