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
from sklearn.metrics import accuracy_score

class TwitterSentimentPrediction:
    def __init__(self):
        pass
        # self.train_labels, self.train_tweet_ids, self.train_tweets = self.read_glove_data("data/train_glove.csv")
        # self.dev_labels, self.dev_tweet_ids, self.dev_tweets = self.read_glove_data("data/dev_glove.csv")
        # self.test_tweet_ids, self.test_tweets = self.read_count_tfidf_data("data/test_count.csv")

        # lr
        # predictions = self.lr_preds(self.train_tweets, self.train_labels, self.dev_tweets)
        # self.write_predictions(self.dev_tweet_ids, predictions, "development/lr_glove_preds.csv")

        # nb
        # self.perform_nb()

        # knn
        # for i in [1, 3, 5, 7]:
        #     testing = self.knn_preds(i, self.train_tweets, self.train_labels, self.dev_tweets)
        #     self.write_predictions(self.dev_tweet_ids, testing, "development/knn_" + str(i) + "_glove_preds.csv")

    # Perform naive bayes for all data sets and write testing
    def perform_nb(self):
        # Count
        train_labels, train_tweet_ids, train_tweets = self.read_count_tfidf_data("data/train_count.csv")
        dev_labels, dev_tweet_ids, dev_tweets = self.read_count_tfidf_data("data/dev_count.csv")
        predictions = self.nb_preds(train_tweets.toarray(), train_labels, dev_tweets.toarray())
        self.write_predictions(dev_tweet_ids, predictions, "development/nb_count_preds.csv")

        # TF-IDF
        train_labels, train_tweet_ids, train_tweets = self.read_count_tfidf_data("data/train_tfidf.csv")
        dev_labels, dev_tweet_ids, dev_tweets = self.read_count_tfidf_data("data/dev_tfidf.csv")
        predictions = self.nb_preds(train_tweets.toarray(), train_labels, dev_tweets.toarray())
        self.write_predictions(dev_tweet_ids, predictions, "development/nb_tfidf_preds.csv")

        # Glove
        train_labels, train_tweet_ids, train_tweets = self.read_glove_data("data/train_glove.csv")
        dev_labels, dev_tweet_ids, dev_tweets = self.read_glove_data("data/dev_glove.csv")
        predictions = self.nb_preds(train_tweets, train_labels, dev_tweets)
        self.write_predictions(dev_tweet_ids, predictions, "development/nb_glove_preds.csv")

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

        lr = LogisticRegression(max_iter=1000).fit(train_data, train_labels)
        predictions = lr.predict(test_data)
        return predictions

    def nb_preds(self, train_data, train_labels, test_data):
        print("Start Naive Bayes...")
        gnb = GaussianNB()
        gnb.fit(train_data, train_labels)
        predictions = gnb.predict(test_data)
        return predictions

    def knn_preds(self, neighbours, train_data, train_labels, test_data):
        print("Start KNN, K = " + str(neighbours) + " ...")
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

    def write_predictions(self, tweet_ids, predictions, file_path):
        print("Writing " + file_path + "...")
        pred_file = open(file_path, "w")
        writer = csv.writer(pred_file)
        writer.writerow(["tweet_id", "sentiment"])
        for i in range(len(tweet_ids)):
            writer.writerow([tweet_ids[i], predictions[i]])
        pred_file.close()
        print("Finish writing " + str(i + 1) + " testing")

    def evaluation(self):
        summary_file = open("development/summary.csv", "w")
        writer = csv.writer(summary_file)
        writer.writerow(["Method", "Accuracy Score"])

        dev_labels, dev_tweet_ids, dev_tweets = self.read_count_tfidf_data("data/dev_count.csv")
        pred_files = ["knn_1_count", "knn_3_count", "knn_5_count", "knn_7_count",
                      "knn_1_tfidf", "knn_3_tfidf", "knn_5_tfidf", "knn_7_tfidf",
                      "knn_1_glove", "knn_3_glove", "knn_5_glove", "knn_7_glove",
                      "nb_count", "nb_tfidf", "nb_glove",
                      "lr_glove"]
        for pred_file in pred_files:
            predictions = pd.read_csv("development/" + pred_file + "_preds.csv")["sentiment"]
            acc_score = accuracy_score(dev_labels, predictions)
            writer.writerow([pred_file, acc_score])

def main():
    tsp = TwitterSentimentPrediction()
    tsp.evaluation()


if __name__ == "__main__":
    main()
