import nltk
import time
import string
from nltk import word_tokenize, pos_tag
from utility import get_directory_content, bcolors
from nltk.corpus import stopwords, wordnet
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter

# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

# Stuff that just need to be initialized one time...
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
global_stopwords = set(nltk.corpus.stopwords.words('english'))
translator = str.maketrans('', '', string.punctuation)

MIN_FREQ = 5

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV

def normalize(token, stem, lemma):
    if (stem):
        token = stemmer.stem(token[0].lower())
    elif (lemma):
        if token[1] is None:
            token = lemmatizer.lemmatize(token[0].lower())
        else:
            token = lemmatizer.lemmatize(token[0].lower(), get_wordnet_pos(token[1]))
    return token
   
def normalize_feelings(feelings, stem, lemma, remove_stopwords, with_tagging):

    start_time = time.time()

    corpus = []
    stopwords = global_stopwords if remove_stopwords else []
    allowed_tags = [wordnet.ADJ, wordnet.VERB, wordnet.NOUN, wordnet.ADV] if with_tagging else []

    for feeling in feelings:

        feeling_without_ponctuation = feeling.translate(translator)

        tokens = word_tokenize(feeling_without_ponctuation)

        tokens_tagged = nltk.pos_tag(tokens)

        tokens = [normalize(token, stem, lemma) for token in tokens_tagged if token[0].lower() not in stopwords and (get_wordnet_pos(token[1]) in allowed_tags or len(allowed_tags) == 0 )]
        
        corpus.extend(tokens)

    #frequencies_words = nltk.FreqDist(corpus)
    #lambda_accepted_tokens = list(map(lambda x : x[0],
    #    filter(lambda x: x[1] >= MIN_FREQ, 
    #        frequencies_words.items())))
    
    # 10 times faster than using FreqDist as sees above: 0.04 vs 0.34
    counter = Counter(corpus)
    accepted_tokens = [key for key in counter.keys() if counter.get(key) >= MIN_FREQ]
    
    filtered_corpus = [token for token in corpus if token in accepted_tokens]

    print("{}# {:.2f} seconds to normalize {}".format(bcolors.WARNING, (time.time() - start_time), bcolors.ENDC))

    return filtered_corpus
    
def main():
    negative_feelings = get_directory_content("Book/neg_Bk/*.text")
    positive_feelings = get_directory_content("Book/pos_Bk/*.text")

    negative_tokens = normalize_feelings(negative_feelings, 
        stem=True, 
        lemma=False, 
        remove_stopwords=True,
        with_tagging=True)
    
    positive_tokens = normalize_feelings(positive_feelings, 
        stem=True, 
        lemma=False, 
        remove_stopwords=True,
        with_tagging=True)
    
    print("fini")

if __name__ == '__main__':  
   main()