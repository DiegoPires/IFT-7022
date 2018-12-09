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

def delete_folder_content(path):
    files = glob.glob(path + '/*')
    for f in files:
        os.remove(f)
    
def clean_results():
    keras_classifiers_path = get_complet_path('results/keras_classifiers')
    sklearn_classifiers_path = get_complet_path('results/sklearn_classifiers')
    results_path = get_complet_path('results')

    delete_folder_content(keras_classifiers_path)
    delete_folder_content(sklearn_classifiers_path)
    delete_folder_content(results_path)

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