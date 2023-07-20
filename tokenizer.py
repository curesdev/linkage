from collections import defaultdict
from hashlib import sha256
from math import exp

import numpy as np



@dataclass
class PIITokenizer:
    first_name: str
    middle_name: str
    last_name: str
    full_name: str
    first_name_soundex: str
    last_name_soundex: str
    gender: str
    date_of_birth: str
    city_at_birth: str
    address_at_bith: str
    state_at_birth: str
    country_at_birth: str








# Tokenize

def tokenize(l, k, eps, fields):
    bf = [0] * l
    for field in fields:
        for i in range(k):
            bf[int(sha256(f"{field}#{i}".encode("utf-8")).hexdigest(), 16) % l] = 1
    eta = 1.0 - 1.0 / (1.0 + exp(eps))
    return np.array([bit if np.random.random() <= eta else 1 - bit for bit in bf], dtype=np.uint8)


# Normalize name


def normalize_name(name):
    return "".join(filter(str.isalpha, name.lower()))


# Q-grams

def q_grams(s, q=2, prefix=""):
    s = normalize_name(s)
    if len(s) < q:
        return [s] + list(s)

    times_seen = defaultdict(int)
    grams = []
    for i in range(len(s) - q + 1):
        gram = s[i : i + q]
        times_seen[gram] += 1
        grams += [f"{prefix}{gram}:{times_seen[gram]}"]

    return grams


def soundex(token):
    """Source: https://www.geeksforgeeks.org/implement-phonetic-search-in-python-with-soundex-algorithm/"""
    token = token.upper()
    soundex = ""
 
    # Retain the First Letter
    soundex += token[0]
 
    # Create a dictionary which maps
    # letters to respective soundex
    # codes. Vowels and 'H', 'W' and
    # 'Y' will be represented by '.'
    dictionary = {"BFPV": "1", "CGJKQSXZ": "2",
                  "DT": "3",
                  "L": "4", "MN": "5", "R": "6",
                  "AEIOUHWY": "."}
 
    # Enode as per the dictionary
    for char in token[1:]:
        for key in dictionary.keys():
            if char in key:
                code = dictionary[key]
                if code != '.':
                    if code != soundex[-1]:
                        soundex += code
 
    # Trim or Pad to make Soundex a
    # 4-character code
    soundex = soundex[:7].ljust(7, "0")
 
    return soundex


def expand_single_value(s):
    return [f"{s}:{x}" for x in range(1, 6)]


def tokenize_name(l, kn, eps, name, prefix):
    expanded = q_grams(name, prefix=prefix)
    k = 1 + kn // len(set(expanded))
    return tokenize(l, k, eps, expanded)


def digit_q_grams(s, q=2, prefix=""):
    s = "".join(filter(str.isdigit, s.lower()))
    if len(s) < q:
        return [s]
    return [prefix + s[i : i + q] for i in range(len(s) - q + 1)]


def tokenize_number(l, kn, eps, name, prefix):
    expanded = digit_q_grams(name, prefix=prefix)
    k = 1 + kn // len(set(expanded))
    return tokenize(l, k, eps, expanded)


def pii_tokenize(l, k, eps, first_name, last_name, city, date_of_birth, gender):
    fields = (
        q_grams(first_name, prefix="first_name:")
        + q_grams(last_name, prefix="last_name:")
        + q_grams(city, prefix="city:")
        + expand_single_value(date_of_birth)
        + expand_single_value(gender)
        # + ["dob:" + date_of_birth, "gender:" + gender]
    )
    bf = tokenize(l, k, eps, fields)
    return "".join(map(str, bf))


def dice(a, b) -> float:
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def similarity(a, b) -> float:
    assert len(a) == len(b)
    return (a == b).sum() / len(a)
