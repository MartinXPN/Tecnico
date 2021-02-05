import logging

from passlib.hash import sha256_crypt
import requests
from hashlib import sha1

from facefive.app import app, mysql

logging.basicConfig(level=logging.DEBUG)

# fetchall, fetchmany(), fetchone()

# INIT DB
def init_db():
    cur = mysql.connection.cursor()
    cur.execute("DROP DATABASE IF EXISTS %s;" % app.config['MYSQL_DB'])
    cur.execute("CREATE DATABASE %s;" % app.config['MYSQL_DB'])
    cur.execute("USE %s;" % app.config['MYSQL_DB'])
    cur.execute("DROP TABLE IF EXISTS Users;")
    cur.execute('''CREATE TABLE Users ( 
                    username VARCHAR(20) NOT NULL,
                    password VARCHAR(100) NOT NULL, 
                    name TEXT,
                    about TEXT, 
                    photo varchar(255) DEFAULT '{}',
                    PRIMARY KEY (username)
                    );'''.format(app.config['default_photo']))
    cur.execute("INSERT INTO Users(username, password, name, about) VALUES (%s, %s, %s, %s)", ('administrator', sha256_crypt.hash('AVeryL33tPasswd'), "Admin", "I have no friends."))
    cur.execute("INSERT INTO Users(username, password, name) VALUES (%s, %s, %s)", ('investor', sha256_crypt.hash('benfica123'), "Mr. Smith"))
    cur.execute("INSERT INTO Users(username, password, name, about) VALUES (%s, %s, %s, %s)", ('ssofadmin', sha256_crypt.hash('SCP'), "SSofAdmin", "A 12-year experienced sys-admin that has developed and secured this application."))
    cur.execute("DROP TABLE IF EXISTS Posts;")
    cur.execute('''CREATE TABLE Posts ( 
                    id int(11) NOT NULL AUTO_INCREMENT,
                    author VARCHAR(20) NOT NULL,
                    content TEXT,
                    type ENUM ('Public','Private','Friends') DEFAULT 'Public',
                    created_at timestamp default now(),
                    updated_at timestamp default now() ON UPDATE now(),
                    PRIMARY KEY (id),
                    FOREIGN KEY (author) REFERENCES Users(username)
                    );''')
    cur.execute("INSERT INTO Posts(author, content, type) VALUES (%s, %s, %s)", ('administrator', 'No one will find that I have no secrets.', "Private"))
    cur.execute("INSERT INTO Posts(author, content, type) VALUES (%s, %s, %s)", ('investor', 'This is a great platform', "Public"))
    cur.execute("INSERT INTO Posts(author, content, type) VALUES (%s, %s, %s)", ('investor', 'Lets keep it for us but I believe that after this app Instagram is done', "Friends"))
    cur.execute("INSERT INTO Posts(author, content, type) VALUES (%s, %s, %s)", ('investor', 'TikTok might also be done but do not want ot make this bold claim in Public', "Private"))
    cur.execute("INSERT INTO Posts(author, content, type) VALUES (%s, %s, %s)", ('SSofAdmin', 'There are no problems with this app. It works perfectly', "Public"))
    cur.execute("INSERT INTO Posts(author, content, type) VALUES (%s, %s, %s)", ('SSofAdmin', 'Cannot put this app running. Can any of my friends help me', "Friends"))
    cur.execute("INSERT INTO Posts(author, content, type) VALUES (%s, %s, %s)", ('SSofAdmin', 'Just found a great new thing. Have a look at it. It might be of help. https://www.guru99.com/install-linux.html', "Public"))
    cur.execute("INSERT INTO Posts(author, content, type) VALUES (%s, %s, %s)", ('SSofAdmin', 'This one is also great. https://www.youtube.com/watch?v=oHg5SJYRHA0&', "Public"))
    cur.execute("DROP TABLE IF EXISTS Friends;")
    cur.execute('''CREATE TABLE Friends ( 
                    id int(11) NOT NULL AUTO_INCREMENT,
                    username1 VARCHAR(20) NOT NULL,
                    username2 VARCHAR(20) NOT NULL,
                    PRIMARY KEY (id),
                    FOREIGN KEY (username1) REFERENCES Users(username),
                    FOREIGN KEY (username2) REFERENCES Users(username)
                    );''')
    cur.execute("INSERT INTO Friends(username1, username2) VALUES (%s, %s)", ('investor', "SSofAdmin"))
    cur.execute("DROP TABLE IF EXISTS FriendsRequests;")
    cur.execute('''CREATE TABLE FriendsRequests ( 
                    id int(11) NOT NULL AUTO_INCREMENT,
                    username1 VARCHAR(20) NOT NULL,
                    username2 VARCHAR(20) NOT NULL,
                    PRIMARY KEY (id),
                    FOREIGN KEY (username1) REFERENCES Users(username),
                    FOREIGN KEY (username2) REFERENCES Users(username)
                    );''')
    cur.execute("INSERT INTO Users(username, password, name, about) VALUES (%s, %s, %s, %s)", ('randomjoe1', sha256_crypt.hash('1'), "Random Joe Smith1", "I am the real Random Joe"))
    cur.execute("INSERT INTO Users(username, password, name) VALUES (%s, %s, %s)", ('randomjoe2', sha256_crypt.hash('2'), "Random Joe Smith2"))
    cur.execute("INSERT INTO Users(username, password, name) VALUES (%s, %s, %s)", ('randomjoe3', sha256_crypt.hash('3'), "Random Joe Smith3"))
    cur.execute("INSERT INTO Users(username, password, name) VALUES (%s, %s, %s)", ('randomjoe4', sha256_crypt.hash('4'), "Random Joe Smith4"))
    cur.execute("INSERT INTO FriendsRequests(username1, username2) VALUES (%s, %s)", ('randomjoe1', "investor"))
    cur.execute("INSERT INTO FriendsRequests(username1, username2) VALUES (%s, %s)", ('randomjoe2', "investor"))
    cur.execute("INSERT INTO FriendsRequests(username1, username2) VALUES (%s, %s)", ('randomjoe3', "investor"))
    cur.execute("INSERT INTO FriendsRequests(username1, username2) VALUES (%s, %s)", ('randomjoe4', "investor"))
   
    mysql.connection.commit()
    cur.close()


# SELECT QUERIES
def get_all_results(q, args=None):
    cur = mysql.connection.cursor()
    cur.execute(q, args=args)
    mysql.connection.commit()
    data = cur.fetchall()
    cur.close()
    return data


# UPDATE and INSERT QUERIES
def commit_results(q, args=None):
    cur = mysql.connection.cursor()
    cur.execute(q, args=args)
    mysql.connection.commit()
    cur.close()


##### Returns a user for a given username
### in: username
### out: User
def get_user(username):
    q = "SELECT * FROM Users WHERE username = %s;"

    logging.debug(f"get_user query: {q % username}")
    data = get_all_results(q, args=(username,))

    if len(data) == 1:
        user = User(*(data[0]))
        return user
    logging.debug(f"get_user: Something wrong happened with (username):{username}")


##### Returns a user for a given pair username:password
### in: username, password
### out: User
def login_user(username, password):
    q = "SELECT * FROM Users WHERE username = %s;"
    
    logging.debug(f"login_user query: {q % username}")
    data = get_all_results(q, args=(username,))

    if len(data) == 1:
        user = User(*(data[0]))
        if sha256_crypt.verify(password, user.password):
            return user
    logging.debug(f"login_user: Something wrong happened with (username, password):({username} {password})")


def weak_password(password):
    sha_hash = sha1(password.encode("utf-8")).hexdigest().upper()
    hash_start = sha_hash[:5]
    hash_end = sha_hash[5:]

    r = requests.get("https://api.pwnedpasswords.com/range/" + hash_start)

    if hash_end in r.text:
        return True
    else:
        return False


##### Registers a new user with a given pair username:password
### in: username, password
### out: User
def register_user(username, password):
    q = "INSERT INTO Users (username, password) VALUES (%s, %s);"

    if weak_password(password):
        raise ValueError("Choose stronger password!")

    password = sha256_crypt.hash(password)
    logging.debug("register_user query: %s" % q)
    commit_results(q, args=(username, password))
    return User(username, password)


##### Updates a user with the given characteristics
### in: username, new_name, new_password, new_about, new_photo
### out: User
def update_user(username, new_name, new_password, new_about, new_photo):
    if new_password != "" and weak_password(new_password):
        raise ValueError("Choose stronger new password")

    new_password = sha256_crypt.hash(new_password)
    q = "UPDATE Users" \
        " SET username=%s, password=%s, name=%s, about=%s, photo=%s" \
        " WHERE username=%s"

    logging.debug(f"update_user query: "
                  f"{q % (username, new_password, new_name, new_about, new_photo, username)}")
    commit_results(q, args=(username, new_password, new_name, new_about, new_photo, username))
    return User(username, new_password, new_name, new_about, new_photo)
    

##### Creates a new post
### in: username, new_content, type
### out: True
def new_post(username, new_content, type):
    q = "INSERT INTO Posts (author, content, type) VALUES (%s, %s, %s)"

    logging.debug(f"new_post query: {q % (username, new_content, type)}")
    commit_results(q, (username, new_content, type))
    return True


##### Gets the post with the given post_id
### in: post_id
### out: Post
def get_post(post_id, username):
    q = "SELECT * FROM Posts WHERE id = %s AND author = %s"

    logging.debug(f"get_post query: {q % (post_id, username)}")
    data = get_all_results(q, args=(post_id, username))

    if len(data) == 1:
        return Post(*(data[0]))
    logging.debug(f"get_post: Something wrong happened with (post_id):{post_id}")


##### Edits the post with the given post_id
### in: post_id, new_content, type
### out: True
def edit_post(post_id, new_content, type, username):
    q = "UPDATE Posts SET content=%s, type=%s WHERE id = %s AND author = %s"

    logging.debug(f"edit_post query: {q % (new_content, type, post_id, username)}")
    commit_results(q, args=(new_content, type, post_id, username))
    return True


##### Returns all posts of a user, from his friends, or public
### in: username
### out: List of Posts_to_show
def get_all_posts(username):
    q = "SELECT Posts.id, Users.username, Users.name, Users.photo, Posts.content, Posts.type, Posts.created_at" \
        " FROM Users INNER JOIN Posts" \
        " ON Users.username = Posts.author" \
        " WHERE Posts.author = %s" \
        " OR (Posts.type = 'Public')" \
        " OR (Posts.type = 'Friends' AND Posts.author IN" \
        " (SELECT username1 from Friends WHERE username2 = %s" \
        "  UNION SELECT username2 from Friends WHERE username1 = %s))"

    logging.debug(f"get_all_posts query: {q % (username, username, username)}")
    data = get_all_results(q, args=(username, username, username))
    posts_to_show = []

    for x in data:
        posts_to_show.append(Post_to_show(*x))

    logging.debug(f"get_all_posts: {posts_to_show}")
    return posts_to_show


##### Creates a new friend request
### in: username (requester), username (new_friend)
### out: True
def new_friend_request(username, new_friend):
    q = "INSERT INTO FriendsRequests (username1, username2) VALUES (%s, %s)"

    logging.debug(f"new_friend_request query: {q % (username, new_friend)}")
    commit_results(q, args=(username, new_friend))
    return True


##### Checks if there is a friend request pending
### in: username (requester), username (new_friend)
### out: data
def is_request_pending(requester, username):
    q = "SELECT username1 FROM FriendsRequests WHERE username1=%s AND username2=%s"
    
    logging.debug(f"is_request_pending query: {q % (requester, username)}")
    data = get_all_results(q, args=(requester, username))
    return data


#### Returns pending friendship requests for the user
### in: username (new_friend)
### out: List of Users
def get_pending_requests(username):
    q = "SELECT * from Users WHERE username IN" \
        " (SELECT username1 FROM FriendsRequests WHERE username2 = %s)"
    
    logging.debug(f"get_pending_requests query: {q % username}")
    data = get_all_results(q, args=(username,))
    users = []

    for x in data:
        users.append(User(*x))

    logging.debug(f"get_pending_requests: {users}")
    return users


##### Accepts a pending friendship requests for the user
### in: username, accept_friend (requester)
### out: True
def accept_friend_request(username, accept_friend):
    q = "INSERT INTO Friends (username1, username2) VALUES (%s, %s);"
    
    logging.debug(f"accept_friend_request query1: {q % (accept_friend, username)}")
    cur = mysql.connection.cursor()
    cur.execute(q, args=(accept_friend, username))

    q = "DELETE FROM FriendsRequests WHERE username1=%s AND username2=%s;"

    logging.debug(f"accept_friend_request query2: {q % (accept_friend, username)}")
    cur.execute(q, args=(accept_friend, username))
    mysql.connection.commit()

    cur.close()
    return True


##### Returns all friends of user that match the search query
### in: username, search_query
### out: List of Users
def get_friends(username, search_query):
    search_query = f'%{search_query}%'
    q = "SELECT * FROM Users WHERE username LIKE %s AND username IN" \
        "       (SELECT username1 FROM Friends WHERE username2 = %s" \
        "  UNION SELECT username2 FROM Friends WHERE username1 = %s)"

    logging.debug(f"get_friends query: {q % (search_query, username, username)}")
    data = get_all_results(q, args=(search_query, username, username))
    friends = []

    for x in data:
        friends.append(User(*x))

    logging.debug(f"get_friends: {friends}")
    return friends


##### Returns the usernames of all friends of user
### in: username
### out: List of usernames
def get_friends_aux(username):
    q = "SELECT username2 FROM Friends WHERE username1 = %s" \
        " UNION " \
        "SELECT username1 FROM Friends WHERE username2 = %s"

    logging.debug(f"get_friends_aux query: {q}")
    data = get_all_results(q, args=(username, username))

    friends = [x[0] for x in data]
    logging.debug(f"get_friends_aux friends: {friends}")
    return friends


##### class User
class User:
    def __init__(self, username, password, name='', about='', photo=''):
        self.username = username
        self.password = password
        self.name = name
        self.about = about
        self.photo = photo
    
    def __repr__(self):
        return '<User: username=%s, password=%s, name=%s, about=%s, photo=%s>' % (self.username, self.password, self.name, self.about, self.photo)


##### class Post
class Post:
    def __init__(self, id, author, content, type, created_at, updated_at):
        self.id = id
        self.author = author
        self.content = content
        self.type = type
        self.created_at = created_at
        self.updated_at = updated_at
    
    def __repr__(self):
        return '<Post: id=%s, author=%s, content=%s, type=%s, created_at=%s, updated_at=%s>' % (self.id, self.author, self.content, self.type, self.created_at, self.updated_at)


##### class Post_to_show (includes Users and Posts information)
class Post_to_show:
    def __init__(self, id, author, name, photo, content, type, created_at):
        self.id = id
        self.author = author
        self.name = name
        self.photo = photo
        self.content = content
        self.type = type
        self.created_at = created_at
        
    def __repr__(self):
        return '<Post_to_show: id=%d, author=%s, name=%s, photo=%s, content=%s, type=%s, created_at=%s>' % (self.id, self.author, self.name, self.photo, self.content, self.type, self.created_at)
