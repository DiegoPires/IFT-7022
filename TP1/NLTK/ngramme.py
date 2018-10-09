import json
import nltk
import collections
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.util import ngrams, bigrams, trigrams
from collections import Counter
from utility import get_complet_path, get_file_content, bcolors
from math import log

# Simply using the most common word between all candidates according to our corpus
def complet_proverbe_with_unigram(tokens, proverbs_to_test):
    print("\n{}## UNIGRAM GUESS{}".format(bcolors.HEADER, bcolors.ENDC))
    
    unigram = list(ngrams(tokens, 1))

    for proverb in proverbs_to_test:
        guess_words = proverbs_to_test[proverb]
    
        filtered_corpus = [x for x in unigram if x[0] in guess_words]

        word_counter = collections.Counter(filtered_corpus)
        best_guess = word_counter.most_common()[0]
        #probability = log(len(unigram) / best_guess[1])

        print(proverb.replace("***",  "{}{}{}").format(bcolors.OKBLUE, best_guess[0][0], bcolors.ENDC))
        #print("P(w): {}{}{} - WORD: {}{}{} \n".format(
        #    bcolors.OKGREEN, probability, bcolors.ENDC,
        #    bcolors.OKBLUE, best_guess[0][0], bcolors.ENDC))

# Using the word before the possibly one as the solution
def complet_proverbe_with_bigram(tokens, proverbs_to_test):
    print("\n{}## BIGRAM GUESS{}".format(bcolors.HEADER, bcolors.ENDC))
    
    bigram = list(bigrams(tokens))
    
    for proverb in proverbs_to_test:
        guess_words = proverbs_to_test[proverb]

        tokenized_proverb = tokenize_text(proverb)
        word_before_unknown_one = tokenized_proverb[tokenized_proverb.index("***")-1]
        
        filtered_corpus = [x for x in bigram if x[0] == word_before_unknown_one and x[1] in guess_words]

        # This is the case that we don't find anything, so came in place the smoothing
        # https://lazyprogrammer.me/probability-smoothing-for-natural-language-processing/
        # need to change the code to calculate all the probability before hand
        if (filtered_corpus):
            word_counter = collections.Counter(filtered_corpus)
            best_guess = word_counter.most_common()[0]
        else:
            best_guess = (('***', '***'), 0)
        
        print(proverb.replace("***",  "{}{}{}").format(bcolors.OKBLUE, best_guess[0][1], bcolors.ENDC))

def complet_proverbe_with_trigram(tokens, proverbs_to_test):
    print("\n{}## TRIGRAM GUESS{}".format(bcolors.HEADER, bcolors.ENDC))
    
    bigram = list(bigrams(tokens))
    
    for proverb in proverbs_to_test:
        guess_words = proverbs_to_test[proverb]

        tokenized_proverb = tokenize_text(proverb)

        # TODO: Cant use the two words before it, the probability would be too low
        # need to use bigram probability of the three possibly options and multiple both probs

def tokenize_text(text):
    ## Custom Regex used instead of the word_tokenizer because it doens't like french that much...
    ## https://stackoverflow.com/questions/47372801/nltk-word-tokenize-on-french-text-is-not-woking-properly
    #return word_tokenize(text, language='french')
    reg_words = r"[dnlt]['´`]|\w+|\$[\d\.]+|\S+"

    tokenizer = RegexpTokenizer(reg_words)
    return tokenizer.tokenize(text)


def main():
    corpus = get_file_content(get_complet_path("proverbes.txt"))
    tests = json.loads(get_file_content(get_complet_path("test1.txt")))

    tokens = tokenize_text(corpus)

    #complet_proverbe_with_unigram(tokens, tests)
    complet_proverbe_with_bigram(tokens, tests)
    #complet_proverbe_with_trigram(tokens, tests)

if __name__ == '__main__':  
   main()