# importing required packages for this section
from urllib.parse import urlparse, urlencode
import ipaddress
import re

# maybe remove if takes too long
# 1.Domain of the URL (Domain)
def getDomain(url):
    domain = urlparse(url).netloc
    if re.match(r"^www.", domain):
        domain = domain.replace("www.", "")

    return domain


# redundent
# 2.Checks for IP address in URL (Have_IP)
def havingIP(url):
    try:
        ipaddress.ip_address(url)
        ip = 1
    except:
        ip = 0
    return ip


# 3.Checks the presence of @ in URL (Have_At)
def haveAtSign(url):
    if "@" in url:
        at = 1
    else:
        at = 0
    return at


# 4.Finding the length of URL and categorizing (URL_Length)
def getLength(url):
    if len(url) < 54:
        length = 0
    else:
        length = 1
    return length


# 5.Gives number of '/' in URL (URL_Depth)
def getDepth(url):
    s = urlparse(url).path.split("/")
    depth = 0
    for j in range(len(s)):
        if len(s[j]) != 0:
            depth = depth + 1
    return depth


# 6.Checking for redirection '//' in the url (Redirection)
def redirection(url):
    pos = url.rfind("//")
    if pos > 6:
        if pos > 7:
            return 1
        else:
            return 0
    else:
        return 0


# 7.Existence of “HTTPS” Token in the Domain Part of the URL (https_Domain)
def httpDomain(url):
    domain = urlparse(url).netloc
    if "https" in domain:
        return 1
    else:
        return 0


# listing shortening services
shortening_services = (
    r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|"
    r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|"
    r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"
    r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|"
    r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|"
    r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|"
    r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|"
    r"tr\.im|link\.zip\.net"
)

# 8. Checking for Shortening Services in URL (Tiny_URL)
def tinyURL(url):
    return re.search(shortening_services, url)


# 9.Checking for Prefix or Suffix Separated by (-) in the Domain (Prefix/Suffix)
def prefixSuffix(url):
    if "-" in urlparse(url).netloc:
        return 1  # phishing
    else:
        return 0  # legitimate


# importing required packages for this section
import re
from bs4 import BeautifulSoup
import whois
import urllib
import urllib.request
from datetime import datetime

# 12.Web traffic (Web_Traffic)
def web_traffic(url):
    try:
        # Filling the whitespaces in the URL if any
        url = urllib.parse.quote(url)
        rank = BeautifulSoup(
            urllib.request.urlopen(
                "http://data.alexa.com/data?cli=10&dat=s&url=" + url
            ).read(),
            "xml",
        ).find("REACH")["RANK"]
        rank = int(rank)
    except Exception as e:
        return 1
    if rank < 100000:
        return 1
    else:
        return 0


# REMOVE
# 14.End time of domain: The difference between termination time and current time (Domain_End)


# importing required packages for this section
import requests

# 15. IFrame Redirection (iFrame)
def iframe(response):
    if response == "":
        return 1
    else:
        if re.findall(r"[<iframe>|<frameBorder>]", response.text):
            return 0
        else:
            return 1


# remove
# 16.Checks the effect of mouse over on status bar (Mouse_Over)
def mouseOver(response):
    if response == "":
        return 1
    else:
        if re.findall("<script>.+onmouseover.+</script>", response.text):
            return 1
        else:
            return 0


# 17.Checks the status of the right click attribute (Right_Click)
def rightClick(response):
    if response == "":
        return 1
    else:
        if re.findall(r"event.button ?== ?2", response.text):
            return 0
        else:
            return 1


# 18.Checks the number of forwardings (Web_Forwards)
def forwarding(response):
    if response == "":
        return 1
    else:
        if len(response.history) <= 2:
            return 0
        else:
            return 1


def create_age_check(num):
    if num <= 12:
        return 1

    else:
        return 0


def expiry_age_check(num):
    if num <= 3:
        return 1

    else:
        return 0


def update_age_check(num):
    if num >= 1000:
        return 1

    else:
        return 0


def exactLength(url):
    length = len(url)
    return length


def zerocount(url):
    num = url.count("0")
    return num

    # 5.11 0 proportion


def zeroprop(url):
    num = url.count("0")
    length = len(url)
    prop = num / length
    return prop


def slashcount(url):
    num = url.count("/")
    return num

    # 5.11 0 proportion


def slashprop(url):
    num = url.count("/")
    length = len(url)
    prop = num / length
    return prop


def percentcount(url):
    num = url.count("%")
    return num

    # 5.11 0 proportion


def percentprop(url):
    num = url.count("%")
    length = len(url)
    prop = num / length
    return prop


def lessthan(url):
    num = url.count(">")
    return num

    # 5.11 0 proportion


def greaterthan(url):
    num = url.count("<")
    return num


# 5.2 count periods
def periodcount(url):
    num = url.count(".")
    return num


# 5.11 0 proportion
def periodprop(url):
    num = url.count(".")
    length = len(url)
    prop = num / length
    return prop


# 5.2 count all special characters according to https://owasp.org/www-community/password-special-characters
def specialcount(url):
    num = 0
    num = num + url.count(" ")
    num = num + url.count("!")
    num = num + url.count('"')
    num = num + url.count("#")
    num = num + url.count("$")
    num = num + url.count("%")
    num = num + url.count("&")
    num = num + url.count("'")
    num = num + url.count("(")
    num = num + url.count(")")
    num = num + url.count("*")
    num = num + url.count("+")
    num = num + url.count(",")
    num = num + url.count("-")
    num = num + url.count(".")
    num = num + url.count("/")
    num = num + url.count(":")
    num = num + url.count(";")
    num = num + url.count("<")
    num = num + url.count("=")
    num = num + url.count(">")
    num = num + url.count("?")
    num = num + url.count("@")
    num = num + url.count("[")
    num = num + url.count("]")
    num = num + url.count("^")
    num = num + url.count("_")
    num = num + url.count("`")
    num = num + url.count("{")
    num = num + url.count("|")
    num = num + url.count("}")
    num = num + url.count("~")
    return num


# prop special
def specialprop(url):
    num = 0
    num = num + url.count(" ")
    num = num + url.count("!")
    num = num + url.count('"')
    num = num + url.count("#")
    num = num + url.count("$")
    num = num + url.count("%")
    num = num + url.count("&")
    num = num + url.count("'")
    num = num + url.count("(")
    num = num + url.count(")")
    num = num + url.count("*")
    num = num + url.count("+")
    num = num + url.count(",")
    num = num + url.count("-")
    num = num + url.count(".")
    num = num + url.count("/")
    num = num + url.count(":")
    num = num + url.count(";")
    num = num + url.count("<")
    num = num + url.count("=")
    num = num + url.count(">")
    num = num + url.count("?")
    num = num + url.count("@")
    num = num + url.count("[")
    num = num + url.count("]")
    num = num + url.count("^")
    num = num + url.count("_")
    num = num + url.count("`")
    num = num + url.count("{")
    num = num + url.count("|")
    num = num + url.count("}")
    num = num + url.count("~")
    length = len(url)
    prop = num / length
    return prop


def check_words(url):
    num = 0
    num = num + url.count("login")
    num = num + url.count("LOGIN")
    num = num + url.count("Login")
    num = num + url.count("registered")
    num = num + url.count("Registered")
    num = num + url.count("Registered")
    return num


# Function to extract features
def featureExtraction(url, label, create_age, expiry_age, update_age):

    features = []
    features.append(getLength(url))
    features.append(getDepth(url))
    features.append(create_age_check(create_age))
    features.append(expiry_age_check(expiry_age))
    features.append(update_age_check(update_age))
    features.append(exactLength(url))
    features.append(zerocount(url))
    features.append(zeroprop(url))
    features.append(periodprop(url))
    features.append(specialcount(url))
    features.append(specialprop(url))
    features.append(slashcount(url))
    features.append(slashprop(url))
    features.append(label)

    return features
