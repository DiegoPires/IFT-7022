import glob
import os
import sys
import re
import numpy as np

# https://stackoverflow.com/questions/287871/print-in-terminal-with-colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def read_file(file):
    path = get_complet_path(file)
    with open(path, encoding = "UTF-8") as f:
        return f.read()

def read_directory(path):
    files = glob.glob(path)
    texts = []
    language = []
    for name in files:
        with open(name, encoding = "ISO-8859-1") as f:
            texts.append(f.read())
            language.append(get_language(name))
    return np.array(language), texts

def get_language(name):
    lg = re.split('-|\.',name)[2]
    if (lg == 'pt'):
        return 'portuguese'
    elif (lg == 'es'):
        return 'spanish'
    elif (lg == 'fr'):
        return 'french'
    elif (lg == 'en'):
        return 'english'

def get_complet_path(path):
    program_dir = os.path.dirname(__file__)
    return os.path.join(program_dir, path)

def get_directory_content(path):
    return read_directory(get_complet_path(path))