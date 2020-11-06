from __future__ import print_function
import sys, os, random
import nltk
import pandas as pd
import time
import re
from sklearn.svm import SVC
from sklearn.linear_model import logistic
from senticlassifier.TrainingTestData import DataUtil
nltk.download('rte')


class Classifier:

    def __init__(self):
        print("Sentiment Classifier Started")

    def get_time_stamp(self):
        return time.strftime("%y%m%d-%H%M%S-%Z")

    def grid(self, alist, blist):
        for a in alist:
            for b in blist:
                yield (a, b)

    NUM_SHOW_FEATURES = 100
    SPLIT_RATIO = 0.9
    FOLDS = 5
    LIST_CLASSIFIERS = ['NaiveBayesClassifier', 'MaxentClassifier', 'SvmClassifier', 'RTEClassifier']

    def trainAndClassify(self, tweets, classifier, fileprefix):

        dataUtil = DataUtil()
        INFO = str(classifier)
        if (len(fileprefix) > 0 and '_' != fileprefix[0]):
            directory = os.path.dirname(fileprefix)
            if not os.path.exists(directory):
                os.makedirs(directory)
            realstdout = sys.stdout
            sys.stdout = open(fileprefix + '_' + INFO + '.txt', 'w')
        print(INFO)
        sys.stderr.write('\n' + '#' * 80 + '\n' + INFO)
        if ('NaiveBayesClassifier' == classifier):
            CLASSIFIER = nltk.classify.NaiveBayesClassifier

            def train_function(v_train):
                return CLASSIFIER.train(v_train)
        elif ('MaxentClassifier' == classifier):
            CLASSIFIER = nltk.classify.MaxentClassifier

            def train_function(v_train):
                return CLASSIFIER.train(v_train, algorithm='GIS', max_iter=10)
        elif ('SvmClassifier' == classifier):
            CLASSIFIER = nltk.classify.scikitlearn.SklearnClassifier(SVC())

            def SvmClassifier_show_most_informative_features(self, n=10):
                print('unimplemented')

            CLASSIFIER.show_most_informative_features = SvmClassifier_show_most_informative_features

            def train_function(v_train):
                return CLASSIFIER.train(v_train)


        elif ('DecisiontreeClassifier' == classifier):
            CLASSIFIER = nltk.classify.DecisionTreeClassifier

            def DecisionTreeClassifier_show_most_informative_features(self, n=10):
                print('unimplemented')

            CLASSIFIER.show_most_informative_features = DecisionTreeClassifier_show_most_informative_features

            def train_function(v_train):
                return CLASSIFIER.train(v_train)

        elif ('RTEClassifier' == classifier):
            CLASSIFIER = nltk.classify.rte_classifier('IIS')

            def train_function(v_train):
                return CLASSIFIER.train(v_train)

        accuracies = []

        for k in range(self.FOLDS):
            (v_train, v_test) = dataUtil.getTrainingAndTestData(tweets, self.FOLDS, k)

            sys.stderr.write('\n[training start]')
            classifier_tot = train_function(v_train)
            sys.stderr.write(' [training complete]')

            print('######################')
            print('1 Step Classifier : ', classifier)
            accuracy_tot = nltk.classify.accuracy(classifier_tot, v_test)
            print('Accuracy : ', accuracy_tot)
            print('######################')
            print(classifier_tot.show_most_informative_features(self.NUM_SHOW_FEATURES))
            print('######################')

            # build confusion matrix over test set
            test_truth = [s for (t, s) in v_test]
            test_predict = [classifier_tot.classify(t) for (t, s) in v_test]

            print('Accuracy :', accuracy_tot)
            print('Confusion Matrix ')
            print(nltk.ConfusionMatrix(test_truth, test_predict))

            accuracies.append(accuracy_tot)
        print("Accuracies:", accuracies)
        print("Average Accuracy:", sum(accuracies) / self.FOLDS)

        sys.stderr.write('\nAccuracies :')
        for k in range(self.FOLDS):
            sys.stderr.write(' %0.5f' % accuracies[k])
        sys.stderr.write('\nAverage Accuracy: %0.5f\n' % (sum(accuracies) / self.FOLDS))
        sys.stderr.flush()

        sys.stdout.flush()
        if (len(fileprefix) > 0 and '_' != fileprefix[0]):
            sys.stdout.close()
            sys.stdout = realstdout

        return classifier_tot

    def main(self, tweets):

        fileprefix = 'logs/run'
        sys.stderr.write('\nlen( tweets ) = ' + str(len(tweets)))
        TIME_STAMP = self.get_time_stamp()
        for cname in self.LIST_CLASSIFIERS:
            self.trainAndClassify(
                tweets, classifier=cname, fileprefix=fileprefix + '_' + TIME_STAMP)