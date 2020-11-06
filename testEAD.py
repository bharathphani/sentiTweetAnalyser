from preprocess import preprocessor
import senticlassifier
# Importing Pandas and NumPy
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#/Users/bhogirala/PycharmProjects/mlws/senti/venv/lib/python2.7/site-packages/pandas/io/parsers.py"
# Importing all datasets
procTweets = pd.read_csv("/Users/bhogirala/PycharmProjects/mlws/senti/data/preprocessed_tweets_test.csv",encoding='ISO-8859-1', index_col=None)

stemmer = nltk.stem.PorterStemmer()
stop_words = set(stopwords.words('english'))
print(stop_words)


all_tweets = []  # DATADICT: all_tweets =   [ (words, sentiment), ... ]
for tuple in procTweets.itertuples():
    # print(tuple[1] + tuple[2] + "\n")
    word_tokens = word_tokenize(tuple[2])
    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    words = [word if (word[0:2] == '__') else word.lower() \
             for word in filtered_sentence \
             if len(word) >= 3]
    words = [stemmer.stem(w) for w in words]  # DATADICT: words = [ 'word1', 'word2', ... ]
    all_tweets.append((words, tuple[1]))

print(all_tweets)

pos_regex = re.compile(r"""(?:
                    ^(?:excellent|wow|awesome|happy|cool|good|love|
                        wonderful|amazing|amaze|bliss|enjoy|fantastic|
                        beautiful|beauty|better|doesnt|fun|funny|arent|luck|lucky|
                        nice|super
                    )$
                )
                |
                n't
                """, re.X)
def get_positive_features(words):
    bag = {}
    for word in words:
        if bool(pos_regex.search(word)):
            key = 'pos(' + word + ')'
            bag[key] = 1


for (words, senti) in all_tweets:
    poswords = get_positive_features(words)
    print(poswords)
