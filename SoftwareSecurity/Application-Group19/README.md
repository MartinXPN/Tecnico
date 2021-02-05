# Fix Report of Group 19

## Phase 2 Deadline: 08Nov2020-23h59m)

_In this template we are using the SQL exercises' image running at `http://mustard.stt.rnl.tecnico.ulisboa.pt:12101/`_

- _Vulnerability 1: SQL Injection in login form allows to login as any user_.
  - Root cause: the source of this vulnerability was the unsafe use of pre-made SQL queries instead of prepared statements.
  - Changes: Changed `login` and `get_all_results` functions to use prepared statements.
<br/><br/>
- _Vulnerability 2: SQL Injection in search my friends form allows to retrieve all the database data_.
  - Root cause: the source of this vulnerability was the unsafe use of pre-made SQL queries instead of prepared statements.
  - Changes: Changed `get_friends` and `get_all_results` functions to use prepared statements.
<br/><br/>
- _Vulnerability 3: Missing user validation allows user to read any stored post_.
  - Root cause: the post was obtained based on the post ``id`` only, instead of also taking into account the user.
  - Changes: Fixed query to take into account the username, changing the ``edit_post (GET)`` and the ``get_post`` functions in view and model, respectively.
<br/><br/>
- _Vulnerability 4: Missing user validation allows user to change any stored post_.
  - Root cause: the post was edited based on the post ``id`` only, instead of also taking into account the user.
  - Changes: Fixed query to take into account the username, changing the ``edit_post (POST)`` and the ``edit_post`` functions in view and model, respectively.
<br/><br/>
- _Vulnerability 5: SQL Injection in create post form allows to retrieve the database data_.
  - Root cause: the source of this vulnerability was unsafe use of pre-made SQL queries.
  - Changes: Changed `new_post` function to use prepared statements.
<br/><br/>
- _Vulnerability 6: SQL Injection in edit post form allows to change any/all post_.
  - Root cause: the source of this vulnerability was the injectable SQL update statement used in ``edit_post (POST)``.
  - Changes: Fixed function `edit_post` by using prepared statements.
<br/><br/>
- _Vulnerability 7: XSS in create post - content of the post can contain script which will be executed on any person viewing the post_.
  - Root cause: the posts were not escaped in the HTML so they could execute a script.
  - Changes: Fixed by turning auto-escape on in `home.html`.
<br/><br/>
- _Vulnerability 8: CSRF in create post_.
  - Root cause: the source of this vulnerability was that the website was vulnerable to CSRF in ``create_post (POST)``.
  - Changes: Fixed by using CSRF tokens.
<br/><br/>
- _Vulnerability 9: XSS in profile page (name, about, photo) - content of the name, about, and photo name can contain script which will be executed on any person viewing profile_.
  - Root cause: the source of this vulnerability was not escaping the HTML so the HTML could execute a script.
  - Changes: Fixed by turning auto-escape on in `friends.html`, `home.html`, and `pending_requests.html`.
<br/><br/>
- _Vulnerability 10: Application vulnerable to DoS (slowloris) attack_.
  - Root cause: the source of this vulnerability was not having a timeout on requests
  - Changes: Fixed by running the app with gunicorn with timeout of several seconds (which can be tuned).
<br/><br/>
- _Vulnerability 11: Stored XSS in photo filename_.
  - Root cause: the source of this vulnerability was not properly handled filename which could contain any characters.
  - Changes: Fixed function `update_profile` making the photo name secure and properly escaped. Also fixed the problem of photo names overlapping for different users.
<br/><br/>
- _Vulnerability 12: Possible DoS by depleting storage space_.
  - Root cause: the source of this vulnerability was saving all the photos and not deleting any afterwards.
  - Changes: Fixed function `update_profile` by deleting the old photo when the new one is uploaded.
<br/><br/>
- _Vulnerability 13: Missing validation for user profile fields name/about/photo_.
  - Root cause: The fields were not escaped and handled properly as they were part of SQL statements and including metacharacters was braking the server. It was also a possible SQLi vulnerability.
  - Changes: Changed `update_user` and `get_all_results` functions to use prepared statements.
<br/><br/>
- _Vulnerability 14: Missing HTTPS encryption_.
  - Root cause: By default the app is configured to run on `http` so the requests are not encrypted.
  - Changes: A fix when deploying the app to production would be to generate `SSL` keys and provide the paths in Dockerfile (commented out).
<br/><br/>
- _Vulnerability 15: Passwords stored in plain text_.
  - Root cause: the source of this vulnerability was that passwords were being stored in the database in plain text.
  - Changes: Fixed function `register_user` by storing instead of the password, its hash using sha256 and in ``login_user`` veryfing the inputted password against the hash.

## Summary
* Use prepared statements in all SQL queries
* Escape HTML content everywhere before displaying
* Run the app with `gunicorn` with timeout and possibly `HTTPS`
* User validation before updating user specific content
* Hash the passwords with `passlib` and require strong passwords with `pwnedpasswords`
* Ensure secure filename for uploaded photos. One photo per user
* Add CSRF token in all the forms

## Notes

- __Use the same numbering that you used for Phase 1.__
- Edit the files directly so that in the final commit there are no vulnerabilities in your application.
- Refer in each commit which vulnerability is fixed.
- Test your own PoCs against the fixed application in order to verify that the vulnerabilities are no longer present (the `assert`s in the end should now fail).
- If you find vulnerabilities that you did not exploit, you may also fix those.
