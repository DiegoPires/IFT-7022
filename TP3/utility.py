import os
import sys
import glob

def read_directory(path):
    files = glob.glob(path)
    texts = []
    for name in files:
        with open(name, encoding = "ISO-8859-1") as f:
            texts.append(f.read())
    return texts

def get_complet_path(path):
    program_dir = os.path.dirname(__file__)
    return os.path.join(program_dir, path)

def get_directory_content(path):
    return read_directory(get_complet_path(path))



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