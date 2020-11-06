Steps to do data setup
1) Goto kaggle  to download the dataset
   https://www.kaggle.com/kazanova/sentiment140
2) Download dataset training.1600000.processed.noemoticon.csv in csv format
3) Rename the dataset to tweets.csv and upload the csv file to data folder in this project
4) Run project run steps mentioned below

Steps to run the project
1) Install dependencies for this project
   install mentioned packages by using pip
   re, ransom, nltk, sys, os, pandas, time, sklearn, ssl
   pip install <package name>
   eg: pip install nltk
2) download RTE, nltk stop words by running installmissing.py in main project folder
3) run main.py file to start training.
4) prerequisites
   1)  Before running main.py make sure data set tweets.csv is uploaded to data folder in main project folder
   2)  After preprocess in main.py file preprocessed_tweets.svc file will get generated
   3)  After random shuffle in main.py preprocessed_tweets_shuffled.csv file will get generated
   4)  After completion and testing and training, a detailed log file will get generated for each classifier in
       logs folder of main project.
5) Added Project_structure.png for reference
5) If any difficulties or issues faced while running thr project please send detailed error log to phani.bharath@gmail.com

