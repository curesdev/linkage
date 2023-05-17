from hashlib import sha256
from math import exp

import numpy as np


def tokenize(l, k, eps, fields):
    bf = [0] * l
    for field in fields:
        for i in range(k):
            bf[int(sha256(f"{field}#{i}".encode("utf-8")).hexdigest(), 16) % l] = 1
    eta = 1.0 - 1.0 / (1.0 + exp(eps))
    return [bit if np.random.random() <= eta else 1 - bit for bit in bf]


def q_grams(s, q=2, prefix=""):
    s = "".join(filter(str.isalpha, s.lower()))
    if len(s) < q:
        return [s]
    return [prefix + s[i : i + q] for i in range(len(s) - q + 1)]


def expand_single_value(s):
    return [f"{s}:{x}" for x in range(1, 6)]


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
    a = np.array(a)
    b = np.array(b)
    return 2 * (a & b).sum() / (a.sum() + b.sum())
