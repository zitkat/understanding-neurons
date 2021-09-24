#!python
# -*- coding: utf-8 -*-
"""
General utils for manipulating data structures, functions and classes.
"""

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

from typing import Dict


def set_key(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = [value]
    elif type(dictionary[key]) == list:
        dictionary[key].append(value)
    else:
        dictionary[key] = [dictionary[key], value]


def append_key(dictionary : Dict, key, value):
    ditem = dictionary.setdefault(key, [])
    if hasattr(ditem, "append"):
        ditem.append(value)
    else:
        dictionary[key] = [ditem, value]
    return dictionary