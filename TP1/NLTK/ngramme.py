import json
import nltk
import collections
import time
import math
from collections import defaultdict
from itertools import permutations
from nltk.tokenize import word_tokenize, RegexpTokenizer
from collections import Counter
from utility import get_complet_path, get_file_content, bcolors, get_file_content_with_br
from math import log

# Create the ngram tuples
def get_ngram(tokens, ngram_count):
    for i in range(len(tokens) - ngram_count+1):
        tuple = ()
        for ngram in range(ngram_count):
            tuple = tuple + (tokens[i+ngram],)
        yield tuple

def tokenize_text(text):
    ## Custom Regex used instead of the word_tokenizer because it doens't like french that much...
    ## https://stackoverflow.com/questions/47372801/nltk-word-tokenize-on-french-text-is-not-woking-properly
    #return word_tokenize(text, language='french')
    reg_words = r"[dnlt]['´`]|\w+|\$[\d\.]+|\S+"

    tokenizer = RegexpTokenizer(reg_words)
    return tokenizer.tokenize(text)

# Simply using the most common word between all candidates according to our corpus
def complet_proverbe_with_unigram(tokens, proverbs_to_test):
    print("\n{}## UNIGRAM GUESS{}".format(bcolors.HEADER, bcolors.ENDC))
    
    unigram = list(get_ngram(tokens, 1))

    for proverb in proverbs_to_test:
        guess_words = proverbs_to_test[proverb]
    
        filtered_corpus = [x for x in unigram if x[0] in guess_words]

        word_counter = collections.Counter(filtered_corpus)
        best_guess = word_counter.most_common()[0]

        print(proverb.replace("***",  "{}{}{}").format(bcolors.OKBLUE, best_guess[0][0], bcolors.ENDC))
        
# Using the word before the possibly one as the solution
def complet_proverbe_with_bigram(tokens, proverbs_to_test):
    print("\n{}## BIGRAM GUESS{}".format(bcolors.HEADER, bcolors.ENDC))
    
    bigram = list(get_ngram(tokens, 2))
    
    for proverb in proverbs_to_test:
        guess_words = proverbs_to_test[proverb]

        tokenized_proverb = tokenize_text(proverb)
        word_before_unknown_one = tokenized_proverb[tokenized_proverb.index("***")-1]
        
        filtered_corpus = [x for x in bigram if x[0] == word_before_unknown_one and x[1] in guess_words]

        if (filtered_corpus):
            word_counter = collections.Counter(filtered_corpus)
            best_guess = word_counter.most_common()[0]
        else:
            best_guess = (('UNK', 'UNK'), 0)
        
        print(proverb.replace("***",  "{}{}{}").format(bcolors.OKBLUE, best_guess[0][1], bcolors.ENDC))

# Using the two words before the possibly one as the solution
def complet_proverbe_with_trigram(tokens, proverbs_to_test):
    print("\n{}## TRIGRAM GUESS{}".format(bcolors.HEADER, bcolors.ENDC))
    
    trigram = list(get_ngram(tokens, 3))

    for proverb in proverbs_to_test:
        guess_words = proverbs_to_test[proverb]

        tokenized_proverb = tokenize_text(proverb)
        word_before_unknown_one = tokenized_proverb[tokenized_proverb.index("***")-1]
        word_before_unknown_one_two = tokenized_proverb[tokenized_proverb.index("***")-2]
        
        filtered_corpus = [x for x in trigram if x[0] == word_before_unknown_one_two and x[1] == word_before_unknown_one and x[2] in guess_words]

        if (filtered_corpus):
            word_counter = collections.Counter(filtered_corpus)
            best_guess = word_counter.most_common()[0]
        else:
            best_guess = (('UNK', 'UNK', 'UNK'), 0)
        
        print(proverb.replace("***",  "{}{}{}").format(bcolors.OKBLUE, best_guess[0][2], bcolors.ENDC))

# Create a table of probabilities with unigram/bigram to be used in the predictions as a 
# progressive way to calculate the probability
def create_table_prob(tokens, ngram_number, laplace):

    start_time = time.time()

    # Trying to create an table with all possibly combinations according with the ngram, but it's to slow, 
    # can't make it to finish
    # for group in permutations(tokens, ngram_number):
    #    quantity = 0
    #    if (laplace > 0):
    #        quantity += laplace
    #    table_prob.append((group, quantity))
    ngram = list(get_ngram(tokens, ngram_number))
    
    ngram_counter = collections.Counter(ngram)
    total = len(tokens)

    if (laplace > 0):
        unique = collections.Counter(tokens)
        total = total + len(unique)

    table_prob = []
    for gram in ngram_counter.most_common():
        quantity = gram[1]
        
        if (ngram_number == 1):
            elem_count = total
        else:
            elem_count = tokens.count(gram[0][0])

        if (laplace > 0):
            quantity += laplace
            elem_count += laplace

        table_prob.append((gram[0], quantity / elem_count))

    print("{}# {:.2f} seconds to create table prod with {} elements{}".format(bcolors.WARNING, (time.time() - start_time), len(table_prob), bcolors.ENDC))

    return table_prob

# Complet phrase with most possible word accordingly to the table of probabilities,
# Unigram and Bigrams (and more grams) are calculate differently
def complet_with_ngram(tokens, proverbs_to_test, ngram_number, laplace=0):
    print("\n{}## NGRAM({}) GUESS | Laplace {}{}".format(bcolors.HEADER, ngram_number, laplace, bcolors.ENDC))

    real_ngram_number = ngram_number
    # Force always to 2 to make a progressive probability, instead of full probability, like done
    # like the complet_proverbe_with_trigram function, for exemple, otherwise probs are too small
    if (ngram_number > 2):
        ngram_number = 2 

    table_prob = create_table_prob(tokens, ngram_number, laplace)

    if (ngram_number == 1):

        for proverb in proverbs_to_test:
            guess_words = proverbs_to_test[proverb]
            
            filtered = [x for x in table_prob if x[0][0] in guess_words]
            filtered.sort(key=lambda elem: elem[1], reverse=True)

            print((proverb.replace("***",  "{}{}{}") + " | Prob {}{}{}").format(
                    bcolors.OKBLUE, 
                    filtered[0][0][0], 
                    bcolors.ENDC,
                    bcolors.OKGREEN,
                    filtered[0][1],
                    bcolors.ENDC))
        
    else:

        for proverb in proverbs_to_test:
            guess_words = proverbs_to_test[proverb]

            guess_word_prob = dict()
            for guess_word in guess_words:

                for i in range(real_ngram_number-1):

                    tokenized_proverb = tokenize_text(proverb[0:proverb.find("***") + 3])
                    count_tokenized_proverb = len(tokenized_proverb) -1

                    search_tuple = (tokenized_proverb[count_tokenized_proverb-i-1], tokenized_proverb[count_tokenized_proverb-i].replace("***", guess_word))

                    bigram_prob = next((x for x in table_prob if x[0][0] == search_tuple[0] and x[0][1] == search_tuple[1]), ("UNK",0.0000001))

                    if  (bigram_prob[0] == "UNK"):
                        # trying to take the real probability of UNK coming after some word, but it never finds :/, so we stay with the real 
                        # small probability for the moment, just in case we don't find anything else
                        #bigram_prob = [x for x in table_prob if x[0][0] == search_tuple[0] and x[0][1] == "UNK"]
                        guess_word = "UNK"

                    if guess_word in guess_word_prob:
                        guess_word_prob[guess_word] *= bigram_prob[1]
                    else:
                        guess_word_prob[guess_word] = bigram_prob[1]
            
            ordered_probs = sorted(guess_word_prob, key=guess_word_prob.get, reverse=True)

            print((proverb.replace("***",  "{}{}{}") + " | Prob {}{}{}").format(
                    bcolors.OKBLUE, 
                    ordered_probs[0], 
                    bcolors.ENDC,
                    bcolors.OKGREEN,
                    guess_word_prob[ordered_probs[0]],
                    bcolors.ENDC))

def probability_of_sentence(tokens, sentence, laplace=0):

    table_prob = create_table_prob(tokens, 2, laplace)
    tokens_of_sentence = tokenize_text(sentence)
    count_tokenized_sentence = len(tokens_of_sentence)

    probability = 0
    for i in range(count_tokenized_sentence-2):

        search_tuple = (tokens_of_sentence[i], tokens_of_sentence[i+1])
        bigram_prob = next((x for x in table_prob if x[0][0] == search_tuple[0] and x[0][1] == search_tuple[1]), ("UNK",0.0001))

        if (probability == 0):
            probability = math.log(bigram_prob[1])
        else:
            probability += math.log(bigram_prob[1])

    return probability

def perplexity_of_sentence(sentence, tokens, laplace=0):

    print("\n{}## PERPLEXITY OF: {} {}".format(bcolors.HEADER, sentence, bcolors.ENDC))

    n = len(collections.Counter(tokens))
        
    probability = probability_of_sentence(tokens, sentence, laplace)
    perplexity = math.pow(2, probability)

    print("# Log Prob is {}{}{} - Perplexity is {}{}{}".format(
        bcolors.OKBLUE, probability, bcolors.ENDC,
        bcolors.OKBLUE, perplexity, bcolors.ENDC))

def main():
    corpus = get_file_content_with_br(get_complet_path("proverbes.txt"))
    tests = json.loads(get_file_content(get_complet_path("test1.txt")))

    tokens = tokenize_text(corpus)
    tokens.append("UNK")

    complet_proverbe_with_trigram(tokens, tests)

    # First solution
    complet_proverbe_with_unigram(tokens, tests)
    complet_proverbe_with_bigram(tokens, tests)
    complet_proverbe_with_trigram(tokens, tests)
    
    # Second solution with laplace
    complet_with_ngram(tokens, tests, ngram_number=1, laplace=1)
    complet_with_ngram(tokens, tests, ngram_number=2, laplace=0)
    
    complet_with_ngram(tokens, tests, ngram_number=3, laplace=0)
    complet_with_ngram(tokens, tests, ngram_number=3, laplace=10)

    # Perplexity
    perplexity_of_sentence("a beau mentir qui vient de loin", tokens, 0)
    perplexity_of_sentence("something not trained in our model", tokens, 0)

if __name__ == '__main__':  
   main()