# Vulnerability 1: SQL Injection in login form allows to login as any user

- Vulnerability: SQL Injection
- Where: `username` in login form
- Impact: Allows access to any users's profile

## Steps to reproduce

1. Insert `username` = `<username>' -- ` and `password` = `any` in login form

[(POC)](vuln1.py)
