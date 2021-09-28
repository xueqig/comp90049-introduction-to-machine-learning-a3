import csv
import math
from ast import literal_eval
import numpy as np
from random import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import pandas as pd
from scipy.sparse import lil_matrix

class TwitterData:
    def __init__(self):
        self.train_labels, self.train_tweet_ids, self.train_tweets = self.read_count_tfidf_data("data/train_tfidf.csv")
        self.test_labels, self.test_tweet_ids, self.test_tweets = self.read_count_tfidf_data("data/test_tfidf.csv")

        # lr
        # predictions = self.lr_preds(self.train_data, self.train_labels, self.test_tweets)
        # self.write_predictions(predictions, "predictions/lr_glove_preds.csv")

        # nb
        # predictions = self.nb_preds(self.train_data, self.train_labels, self.test_tweets)
        # self.write_predictions(predictions, "predictions/nb_glove_preds.csv")

        # knn
        predictions = self.knn_preds(7, self.train_tweets, self.train_labels, self.test_tweets)
        self.write_predictions(predictions, "predictions/knn_7_tfidf_preds.csv")

    # Read count and tfidf data
    def read_count_tfidf_data(self, file_path):
        data = pd.read_csv(file_path, dtype={"sentiment": str, "tweet_id": int}, converters={"tweet": literal_eval})
        labels = np.array(list(data["sentiment"]))
        tweet_ids = np.array(list(data["tweet_id"]))
        tweets = list(data["tweet"])

        tweets_sparse_matrix = lil_matrix((len(tweets), 5000))

        for i in range(len(tweets)):
            tweet = tweets[i]
            for j in range(len(tweet)):
                word = tweet[j]
                tweets_sparse_matrix[i, word[0]] = word[1]

        return labels, tweet_ids, tweets_sparse_matrix

    # Read glove data
    def read_glove_data(self, file_path):
        data = pd.read_csv(file_path, dtype={"sentiment": str, "tweet_id": int}, converters={"tweet": literal_eval})
        labels = list(data["sentiment"])
        tweet_ids = list(data["tweet_id"])
        tweets = list(data["tweet"])
        return labels, tweet_ids, tweets

    def read_raw_data(self, file_path):
        data = pd.read_csv(file_path, dtype={"sentiment": str, "tweet_id": int, "tweet": str})
        labels = np.array(list(data["sentiment"]))
        tweet_ids = np.array(list(data["tweet_id"]))
        tweets = np.array(list(data["tweet"]))

        return labels, tweet_ids, tweets

    def lr_preds(self, train_data, train_labels, test_data):
        scaler = preprocessing.StandardScaler().fit(train_data)
        train_data = scaler.transform(train_data)

        scaler = preprocessing.StandardScaler().fit(test_data)
        test_data = scaler.transform(test_data)

        lr = LogisticRegression().fit(train_data, train_labels)
        predictions = lr.predict(test_data)
        return predictions

    def nb_preds(self, train_data, train_labels, test_data):
        gnb = GaussianNB()
        gnb.fit(train_data, train_labels)
        predictions = gnb.predict(test_data)
        return predictions

    def knn_preds(self, neighbours, train_data, train_labels, test_data):
        print("knn")
        knc = KNeighborsClassifier(n_neighbors=neighbours)
        knc.fit(train_data, train_labels)
        predictions = knc.predict(test_data)
        return predictions

    def rand_preds(self, test_tweet_ids):
        labels = ["pos", "neu", "neg"]
        rand_preds = []
        for i in range(len(test_tweet_ids)):
            rand_idx = math.floor(len(labels) * random())
            rand_preds.append(labels[rand_idx])
        return rand_preds

    def weight_rand_preds(self, train_labels, test_tweet_ids):
        train_len = len(train_labels)
        train_pos_pct = train_labels.count("pos") / train_len
        train_neu_pct = train_labels.count("neu") / train_len
        train_neg_pct = train_labels.count("neg") / train_len

        test_len = len(test_tweet_ids)
        test_pos = math.ceil(test_len * train_pos_pct)
        test_neu = math.ceil(test_len * train_neu_pct)
        test_neg = test_len - test_pos - test_neu

        weight_rand_preds = []

        # Insert pos
        for i in range(test_pos):
            weight_rand_preds.append("pos")

        # Choose random places to insert neu
        for i in range(test_neu):
            rand_idx = int(len(weight_rand_preds) * random())
            weight_rand_preds = weight_rand_preds[:rand_idx] + ["neu"] + weight_rand_preds[rand_idx:]

        # Choose random places to insert neg
        for i in range(test_neg):
            rand_idx = int(len(weight_rand_preds) * random())
            weight_rand_preds = weight_rand_preds[:rand_idx] + ["neg"] + weight_rand_preds[rand_idx:]

        return weight_rand_preds

    def write_predictions(self, predictions, file_path):
        print("Writing " + file_path + "...")
        pred_file = open(file_path, "w")
        writer = csv.writer(pred_file)
        writer.writerow(["tweet_id", "sentiment"])
        for i in range(len(self.test_tweet_ids)):
            writer.writerow([self.test_tweet_ids[i], predictions[i]])
        pred_file.close()
        print("Finish writing " + str(i + 1) + " predictions")


def main():
    td = TwitterData()


if __name__ == "__main__":
    main()
