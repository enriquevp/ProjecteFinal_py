import pandas as pd
import numpy as nb
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords

def load_data():
    df = pd.read_csv("../data/twitter-sentiment-dataset.csv", delimiter=",")
    df_test = df[1301:2600]
    df_train = df[:1300]
    return (df_train, df_test)

def get_tweets_words(df):
    all_words = []
    for index, row in df.iterrows():
        for word in word_tokenize(row['SentimentText']):
            all_words.append(word)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

def get_tweet_and_sentiment(df):
    tweets = []
    for index, row in df.iterrows():
        current_tweet = []
        for word in word_tokenize(row['SentimentText']):
            current_tweet.append(word)
        tweets.append((current_tweet, row['Sentiment']))
    return tweets

def extract_features(doc, filtered_features):
    doc_words = set(doc)
    features = {}
    for word in filtered_features:
        features['contains(%s)' % word] = (word in doc_words)
    return features
