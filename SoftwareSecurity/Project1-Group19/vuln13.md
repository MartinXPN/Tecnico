# Vulnerability 13: Missing validation for user profile fields name/about/photo

- Vulnerability: Missing validation of the input
- Where: `/profile` in `about`, `name`, `photo name` no input validation
- Impact: Breaks the website and possible SQL Injection as 
    those fields are part of an sql statement

## Steps to reproduce
1. Register and login a user
2. Open `/profile` page
3. Edit one of the fields `about`, `name`, `photo name` that contain `'` character.
    That will break the SQL statement of which those fields are part of.

[(POC)](vuln13.py)
