from __future__ import print_function
from senticlassifier.SentimentClassifier import Classifier
from preprocess import preprocessor
import pandas as pd
import random
print("reading csv")
tweet_data = pd.read_csv("./data/tweets.csv",encoding='ISO-8859-1',names=["label", "id", "date", "query_flag", "user", "tweet"])
tweets = tweet_data['tweet']
print("pre-processing started")
preprocess = preprocessor.Preprocessor()
tweets_1 = tweets.apply(lambda x: preprocess.processAll(x))
tweetdf = pd.DataFrame({'sentiment': tweet_data['label'],
                   'tweet': tweets_1})
tweetdf.to_csv("./data/preprocessed_tweets.csv" ,index=False)
fid = open("./data/preprocessed_tweets.csv", "r")
li = fid.readlines()
fid.close()
print("shuffling started")
random.shuffle(li)
fid1 = open("./data/preprocessed_tweets_shuffled.csv", "w")
fid1.writelines(li)
fid1.close()
print("shuffled")
classifier = Classifier()
print("Calling main function")
tweets = pd.read_csv("./data/preprocessed_tweets_shuffled.csv",
                             encoding='ISO-8859-1', names=["label", "id", "date", "query", "user", "tweet"])
classifier.main(tweets)