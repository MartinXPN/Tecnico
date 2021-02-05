var pos = document.URL.indexOf("name=") + 5;
var name = document.URL.substring(pos, document.URL.length);
var sanitizedName = encodeURI(name);
document.write(sanitizedName);
