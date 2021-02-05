# Vulnerability 12: Possible DoS by depleting storage space

- Vulnerability: DoS
- Where: `photo` in `POST /update_profile`
- Impact: Server stores files of any type and size. Old pictures are stored even after the change. The possibility of running out of disc space exists.

## Steps to reproduce

1. Register a new user (if you already have an user you can go to step 2).
2. Login as that user.
3. Update profile uploading any picture as `photo`.
4. Stored picture is available on `/static/photos/xxxxxxxx_fileName`.
5. After uploading a new picture the same way, the link to the first uploaded picture is still valid.

[(POC)](vuln12.py)
