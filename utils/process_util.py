#!python
# -*- coding: utf-8 -*-
"""
Utils for managing processing.
"""

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

import logging
import time
from pathlib import Path

plogger = logging.getLogger('training_pipeline')
plogger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
plogger.addHandler(ch)


def add_plog_file(file_path):
    process_log_file = ensured_path(file_path)
    fh = logging.FileHandler(process_log_file)
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    plogger.addHandler(fh)


def make_path(*pathargs, isdir=False, **pathkwargs):
    new_path = Path(*pathargs, **pathkwargs)
    return ensured_path(new_path, isdir=isdir)


def ensured_path(path: Path, isdir=False):
    if isdir:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def now():
    """
    :return: date and time as YYYY-mm-dd-hh-MM
    """
    return time.strftime("%Y-%m-%d-%H-%M")