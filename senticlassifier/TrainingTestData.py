import pandas as pd
import nltk
import sys
import  re
from nltk.corpus import stopwords


class DataUtil :

    def getTrainingAndTestData(self, tweets, K, k):

        from functools import wraps
        #procTweets1 = pd.read_csv("/Users/bhogirala/PycharmProjects/mlws/senti/data/preprocessed_tweets_shuffled.csv",encoding='ISO-8859-1')
        procTweets = tweets[:10]
        stemmer = nltk.stem.PorterStemmer()

        all_tweets = []  # DATADICT: all_tweets =   [ (words, sentiment), ... ]
        for tuple in procTweets.itertuples():
            # print(tuple[1] + tuple[2] + "\n")
            words = [word if (word[0:2] == '__') else word.lower() \
                     for word in tuple[2].split() \
                     if len(word) >= 3]
            words = [stemmer.stem(w) for w in words]  # DATADICT: words = [ 'word1', 'word2', ... ]
            all_tweets.append((words, tuple[1]))

        train_tweets = [x for i, x in enumerate(all_tweets) if i % K != k]
        test_tweets = [x for i, x in enumerate(all_tweets) if i % K == k]


        def get_word_features(words):
            bag = {}
            stop_words = set(stopwords.words('english'))
            filtered_words = [w for w in words if not w in stop_words]
            words_uni = ['has(%s)' % ug for ug in filtered_words]
            for f in words_uni:
                bag[f] = 1

            # bag = collections.Counter(words_uni+words_bi+words_tri)
            return bag

        negtn_regex = re.compile(r"""(?:
            ^(?:never|no|nothing|nowhere|noone|none|not|
                havent|hasnt|hadnt|cant|couldnt|shouldnt|
                wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint
            )$
        )
        |
        n't
        """, re.X)

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

        def get_negation_features(words):
            INF = 0.0
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

        def get_positive_features(words):

            bag={}
            for word in words:
                if bool(pos_regex.search(word)):
                    key = 'pos(' + word + ')'
                    bag[key] = 1
            return bag


        def counter(func):  # http://stackoverflow.com/questions/13512391/to-count-no-times-a-function-is-called
            @wraps(func)
            def tmp(*args, **kwargs):
                tmp.count += 1
                return func(*args, **kwargs)

            tmp.count = 0
            return tmp

        @counter  # http://stackoverflow.com/questions/13512391/to-count-no-times-a-function-is-called
        def extract_features(words):

            features = {}
            negation_features = get_negation_features(words)
            features.update(negation_features)
            postive_features = get_positive_features(words)
            features.update(postive_features)
            word_features = get_word_features(words)
            features.update(word_features)
            sys.stderr.write('\rfeatures extracted for ' + str(extract_features.count) + ' tweets')
            return features

        extract_features.count = 0;
        tweets_processed = 0
        # Apply NLTK's Lazy Map
        print("length of train tweets "+str(len(train_tweets)))
        v_train = nltk.classify.apply_features(extract_features, train_tweets)
        print("length of test tweets " + str(len(test_tweets)))
        v_test = nltk.classify.apply_features(extract_features, test_tweets)
        return (v_train, v_test)