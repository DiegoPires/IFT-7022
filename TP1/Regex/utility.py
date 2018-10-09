import os
import sys

def get_file_content(path):
    bag_of_text = {}
    count = 0
    with open(path) as fp:
        for line in fp:
            bag_of_text[count] = line.strip()
            count+=1
    return bag_of_text

def get_complet_path(path):
    program_dir = os.path.dirname(__file__)
    return os.path.join(program_dir, path)
