import pandas as pd
import os

def get_complet_path(path):
    program_dir = os.path.dirname(__file__)
    return os.path.join(program_dir, path)

def main(verbose=False):

    df = pd.read_table(get_complet_path('data/train.txt'), sep='\t', header=0, usecols=[1,2,3,4])
    df.fillna('', inplace=True)
    print(df)
    

if __name__ == '__main__':  
   main(verbose=False)