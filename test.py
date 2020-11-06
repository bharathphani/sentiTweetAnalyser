from preprocess import preprocessor
import senticlassifier
# Importing Pandas and NumPy
import pandas as pd
import nltk
import re

#/Users/bhogirala/PycharmProjects/mlws/senti/venv/lib/python2.7/site-packages/pandas/io/parsers.py"
# Importing all datasets
procTweet = pd.read_csv("/Users/bhogirala/PycharmProjects/mlws/senti/data/preprocessed_tweets_test.csv",encoding='ISO-8859-1', index_col=None)

cols = list(procTweet)
stemmer = nltk.stem.PorterStemmer()
all_tweets = []

procTweet_short = procTweet[0:1]

for tuple in procTweet_short.itertuples():
         #print(tuple[1] + tuple[2] + "\n")
         words = [word if (word[0:2] == '--') else word.lower() \
                  for word in tuple[2].split() \
                  if len(word) >= 3]
         words = [stemmer.stem(w) for w in words]  # DATADICT: words = [ 'word1', 'word2', ... ]
         all_tweets.append((words, tuple[1]))

#print(all_tweets)

unigrams_fd = nltk.FreqDist()
bi_grams_fd = nltk.FreqDist()
tri_grams_fd = nltk.FreqDist()
for (words, sentiment) in all_tweets:
    words_uni = words
    unigrams_fd.update(words)
    words_bi = [','.join(map(str, bg)) for bg in nltk.bigrams(words)]
    bi_grams_fd.update(words_bi)
    words_tri = [','.join(map(str, tg)) for tg in nltk.trigrams(words)]
    tri_grams_fd.update(words_tri)


def get_negation_features(words):
    INF = 0.0
    negtn_regex = re.compile(r"""(?:
                ^(?:never|no|nothing|nowhere|noone|none|not|
                    havent|hasnt|hadnt|cant|couldnt|shouldnt|
                    wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint
                )$
            )
            |
            n't
            """, re.X)
    negtn = [bool(negtn_regex.search(w)) for w in words]

    left = [0.0] * len(words)
    prev = 0.0
    for i in range(0, len(words)):
        if (negtn[i]):
            prev = 1.0
        left[i] = prev
        prev = max(0.0, prev - 0.1)

    right = [0.0] * len(words)
    prev = 0.0
    for i in reversed(range(0, len(words))):
        if (negtn[i]):
            prev = 1.0
        right[i] = prev
        prev = max(0.0, prev - 0.1)

    return dict(zip(
        ['neg_l(' + w + ')' for w in words] + ['neg_r(' + w + ')' for w in words],
        left + right))
features = {}

for (words, sentiment) in all_tweets:
    bag = {}
    words_uni = ['has(%s)' % ug for ug in words]

    words_bi = ['has(%s)' % ','.join(map(str, bg)) for bg in nltk.bigrams(words)]

    words_tri = ['has(%s)' % ','.join(map(str, tg)) for tg in nltk.trigrams(words)]

    #print(words_uni) #eg: has(__hndl) has(that) etc
    #print(words_bi) #eg: has(dive,mani), has(mani,time)
    #print(words_tri) #eg:  has(mani,time,for), has(time,for,the)

    for f in words_uni + words_bi + words_tri:
        bag[f] = 1

    #print(bag)  #['has(__hndl)' : 1, 'has(dive,mani)': 1,  'has(mani,time,for): 1']

    neg_features = get_negation_features(words)
    #print(neg_features) #neg_l(__hndl)': 0.0 , neg_l(behav)': 0.9, neg_l(cant)': 1.0


    features.update(bag)
    features.update(neg_features)

v_train = nltk.classify.apply_features(features, all_tweets)
print(features)











