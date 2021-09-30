import math
from random import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from DataProcessing import TwitterDataProcessing


class SentimentPrediction:
    def __init__(self):
        self.tdp = TwitterDataProcessing()
        self.train_labels_count, self.train_tweet_ids_count, self.train_tweets_count = self.tdp.read_count_tfidf_data("data/train_count.csv")
        self.dev_labels_count, self.dev_tweet_ids_count, self.dev_tweets_count = self.tdp.read_count_tfidf_data("data/dev_count.csv")

        self.train_labels_tfidf, self.train_tweet_ids_tfidf, self.train_tweets_tfidf = self.tdp.read_count_tfidf_data("data/train_tfidf.csv")
        self.dev_labels_tfidf, self.dev_tweet_ids_tfidf, self.dev_tweets_tfidf = self.tdp.read_count_tfidf_data("data/dev_tfidf.csv")

        self.train_labels_glove, self.train_tweet_ids_glove, self.train_tweets_glove = self.tdp.read_glove_data("data/train_glove.csv")
        self.dev_labels_glove, self.dev_tweet_ids_glove, self.dev_tweets_glove = self.tdp.read_glove_data("data/dev_glove.csv")

    def dt_predictions(self):
        # Count
        train_labels, train_tweet_ids, train_tweets = self.tdp.read_count_tfidf_data("data/train_count.csv")
        dev_labels, dev_tweet_ids, dev_tweets = self.tdp.read_count_tfidf_data("data/dev_count.csv")
        predictions = self.decision_tree(train_tweets, train_labels, dev_tweets)
        self.tdp.write_predictions(dev_tweet_ids, predictions, "development/dt_count_preds.csv")

        # TF-IDF
        train_labels, train_tweet_ids, train_tweets = self.tdp.read_count_tfidf_data("data/train_tfidf.csv")
        dev_labels, dev_tweet_ids, dev_tweets = self.tdp.read_count_tfidf_data("data/dev_tfidf.csv")
        predictions = self.decision_tree(train_tweets, train_labels, dev_tweets)
        self.tdp.write_predictions(dev_tweet_ids, predictions, "development/dt_tfidf_preds.csv")

        # Glove
        train_labels, train_tweet_ids, train_tweets = self.tdp.read_glove_data("data/train_glove.csv")
        dev_labels, dev_tweet_ids, dev_tweets = self.tdp.read_glove_data("data/dev_glove.csv")
        predictions = self.decision_tree(train_tweets, train_labels, dev_tweets)
        self.tdp.write_predictions(dev_tweet_ids, predictions, "development/dt_glove_preds.csv")

    # neural_network
    def nn_predictions(self):
        # Count
        predictions_count = self.neural_network(self.train_tweets_count, self.train_labels_count, self.dev_tweets_count)
        self.tdp.write_predictions(self.dev_tweet_ids_count, predictions_count, "development/nn_count_preds.csv")

        # TF-IDF
        predictions_tfidf = self.neural_network(self.train_tweets_tfidf, self.train_labels_tfidf, self.dev_tweets_tfidf)
        self.tdp.write_predictions(self.dev_tweet_ids_tfidf, predictions_tfidf, "development/nn_tfidf_preds.csv")

        # Glove
        predictions_glove = self.neural_network(self.train_tweets_glove, self.train_labels_glove, self.dev_tweets_glove)
        self.tdp.write_predictions(self.dev_tweet_ids_glove, predictions_glove, "development/nn_glove_preds.csv")

    # Perform naive bayes for all data sets and write testing
    def nb_predictions(self):
        # Count
        train_labels, train_tweet_ids, train_tweets = self.tdp.read_count_tfidf_data("data/train_count.csv")
        dev_labels, dev_tweet_ids, dev_tweets = self.tdp.read_count_tfidf_data("data/dev_count.csv")
        predictions = self.naive_bayes(train_tweets.toarray(), train_labels, dev_tweets.toarray())
        self.tdp.write_predictions(dev_tweet_ids, predictions, "development/nb_count_preds.csv")

        # TF-IDF
        train_labels, train_tweet_ids, train_tweets = self.tdp.read_count_tfidf_data("data/train_tfidf.csv")
        dev_labels, dev_tweet_ids, dev_tweets = self.tdp.read_count_tfidf_data("data/dev_tfidf.csv")
        predictions = self.naive_bayes(train_tweets.toarray(), train_labels, dev_tweets.toarray())
        self.tdp.write_predictions(dev_tweet_ids, predictions, "development/nb_tfidf_preds.csv")

        # Glove
        train_labels, train_tweet_ids, train_tweets = self.tdp.read_glove_data("data/train_glove.csv")
        dev_labels, dev_tweet_ids, dev_tweets = self.tdp.read_glove_data("data/dev_glove.csv")
        predictions = self.naive_bayes(train_tweets, train_labels, dev_tweets)
        self.tdp.write_predictions(dev_tweet_ids, predictions, "development/nb_glove_preds.csv")

    def lr_predictions(self):
        train_labels_count, train_tweet_ids_count, train_tweets_count = self.tdp.read_count_tfidf_data("data/train_count.csv")
        dev_labels_count, dev_tweet_ids_count, dev_tweets_count = self.tdp.read_count_tfidf_data("data/dev_count.csv")
        predictions = self.logistic_regression(train_tweets_count, train_labels_count, dev_tweets_count)
        self.tdp.write_predictions(dev_tweet_ids_count, predictions, "development/lr_count_preds.csv")

        train_labels_tfidf, train_tweet_ids_tfidf, train_tweets_tfidf = self.tdp.read_count_tfidf_data("data/train_tfidf.csv")
        dev_labels_tfidf, dev_tweet_ids_tfidf, dev_tweets_tfidf = self.tdp.read_count_tfidf_data("data/dev_tfidf.csv")
        predictions = self.logistic_regression(train_tweets_tfidf, train_labels_tfidf, dev_tweets_tfidf)
        self.tdp.write_predictions(dev_tweet_ids_tfidf, predictions, "development/lr_tfidf_preds.csv")

        train_labels_glove, train_tweet_ids_glove, train_tweets_glove = self.tdp.read_glove_data("data/train_glove.csv")
        dev_labels_glove, dev_tweet_ids_glove, dev_tweets_glove = self.tdp.read_glove_data("data/dev_glove.csv")
        predictions = self.logistic_regression(train_tweets_glove, train_labels_glove, dev_tweets_glove)
        self.tdp.write_predictions(dev_tweet_ids_glove, predictions, "development/lr_glove_preds.csv")

    def decision_tree(self, train_tweet, train_labels, test_tweet):
        print("Start Decision Tree...")
        dtc = DecisionTreeClassifier(criterion="entropy")
        dtc.fit(train_tweet, train_labels)
        predictions = dtc.predict(test_tweet)
        return predictions

    def neural_network(self, train_tweet, train_labels, test_tweet):
        print("Start Neural Network...")
        mlpc = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5, 2), random_state = 1)
        mlpc.fit(train_tweet, train_labels)
        predictions = mlpc.predict(test_tweet)
        return predictions

    def logistic_regression(self, train_tweet, train_labels, test_tweet):
        print("Start Logistic Regression...")
        lr = LogisticRegression(max_iter=1000).fit(train_tweet, train_labels)
        predictions = lr.predict(test_tweet)
        return predictions

    def naive_bayes(self, train_data, train_labels, test_data):
        print("Start Naive Bayes...")
        gnb = GaussianNB()
        gnb.fit(train_data, train_labels)
        predictions = gnb.predict(test_data)
        return predictions

    def k_neighbors(self, neighbours, train_data, train_labels, test_data):
        print("Start KNN, K = " + str(neighbours) + " ...")
        knc = KNeighborsClassifier(n_neighbors=neighbours)
        knc.fit(train_data, train_labels)
        predictions = knc.predict(test_data)
        return predictions

    def random_baseline(self, test_tweet_ids):
        labels = ["pos", "neu", "neg"]
        rand_preds = []
        for i in range(len(test_tweet_ids)):
            rand_idx = math.floor(len(labels) * random())
            rand_preds.append(labels[rand_idx])
        return rand_preds

    def weighted_random_baseline(self, train_labels, test_tweet_ids):
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


def main():
    sp = SentimentPrediction()
    sp.nn_predictions()


if __name__ == "__main__":
    main()
