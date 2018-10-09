import os
import sys

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

def get_file_content(path):
    text = ""
    with open(path) as fp:
        for line in fp:
            text += line.strip()
    return text

def get_complet_path(path):
    program_dir = os.path.dirname(__file__)
    return os.path.join(program_dir, path)
