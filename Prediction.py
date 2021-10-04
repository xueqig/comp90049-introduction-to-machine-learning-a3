import math
from random import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from DataProcessing import TwitterDataProcessing
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler


class SentimentPrediction:
    def __init__(self):
        self.tdp = TwitterDataProcessing()

        self.train_labels_count, self.train_tweet_ids_count, self.train_tweets_count = self.tdp.read_count_tfidf_data("data/train_count.csv")
        self.dev_labels_count, self.dev_tweet_ids_count, self.dev_tweets_count = self.tdp.read_count_tfidf_data("data/dev_count.csv")

        self.train_labels_tfidf, self.train_tweet_ids_tfidf, self.train_tweets_tfidf = self.tdp.read_count_tfidf_data("data/train_tfidf.csv")
        self.dev_labels_tfidf, self.dev_tweet_ids_tfidf, self.dev_tweets_tfidf = self.tdp.read_count_tfidf_data("data/dev_tfidf.csv")

        self.train_labels_glove, self.train_tweet_ids_glove, self.train_tweets_glove = self.tdp.read_glove_data("data/train_glove.csv")
        self.dev_labels_glove, self.dev_tweet_ids_glove, self.dev_tweets_glove = self.tdp.read_glove_data("data/dev_glove.csv")

    def knn_predictions(self, neighbours):
        # Count
        predictions_count = self.k_neighbors(neighbours, self.train_tweets_count, self.train_labels_count, self.dev_tweets_count)
        self.tdp.write_predictions(self.dev_tweet_ids_count, predictions_count, "development/knn_" + str(neighbours) + "_count_preds.csv")

        # TF-IDF
        predictions_tfidf = self.k_neighbors(neighbours, self.train_tweets_tfidf, self.train_labels_tfidf, self.dev_tweets_tfidf)
        self.tdp.write_predictions(self.dev_tweet_ids_tfidf, predictions_tfidf, "development/knn_" + str(neighbours) + "_tfidf_preds.csv")

        # Glove
        predictions_glove = self.k_neighbors(neighbours, self.train_tweets_glove, self.train_labels_glove, self.dev_tweets_glove)
        self.tdp.write_predictions(self.dev_tweet_ids_glove, predictions_glove, "development/knn_" + str(neighbours) + "_glove_preds.csv")

    def dt_predictions(self, max_depth):
        # Count
        predictions_count = self.decision_tree(max_depth, self.train_tweets_count, self.train_labels_count, self.dev_tweets_count)
        self.tdp.write_predictions(self.dev_tweet_ids_count, predictions_count, "development/dt_" + str(max_depth) + "_count_preds.csv")

        # TF-IDF
        predictions_tfidf = self.decision_tree(max_depth, self.train_tweets_tfidf, self.train_labels_tfidf, self.dev_tweets_tfidf)
        self.tdp.write_predictions(self.dev_tweet_ids_tfidf, predictions_tfidf, "development/dt_" + str(max_depth) + "_tfidf_preds.csv")

        # Glove
        predictions_glove = self.decision_tree(max_depth, self.train_tweets_glove, self.train_labels_glove, self.dev_tweets_glove)
        self.tdp.write_predictions(self.dev_tweet_ids_glove, predictions_glove, "development/dt_" + str(max_depth) + "_glove_preds.csv")

    # neural_network
    def nn_predictions(self):
        # Count
        predictions_count = self.neural_network(self.train_tweets_count, self.train_labels_count, self.dev_tweets_count)
        self.tdp.write_predictions(self.dev_tweet_ids_count, predictions_count, "development/nn_64_count_preds.csv")

        # TF-IDF
        predictions_tfidf = self.neural_network(self.train_tweets_tfidf, self.train_labels_tfidf, self.dev_tweets_tfidf)
        self.tdp.write_predictions(self.dev_tweet_ids_tfidf, predictions_tfidf, "development/nn_64_tfidf_preds.csv")

        # Glove
        predictions_glove = self.neural_network(self.train_tweets_glove, self.train_labels_glove, self.dev_tweets_glove)
        self.tdp.write_predictions(self.dev_tweet_ids_glove, predictions_glove, "development/nn_64_glove_preds.csv")

    # Perform naive bayes for all data sets and write testing
    def multinomial_nb_predictions(self):
        print("Start MultinomialNB on count data...")
        train_labels, train_tweet_ids, train_tweets = self.tdp.read_count_tfidf_data("data/train_count.csv")
        dev_labels, dev_tweet_ids, dev_tweets = self.tdp.read_count_tfidf_data("data/dev_count.csv")

        clf = MultinomialNB()
        clf.fit(train_tweets, train_labels)
        predictions = clf.predict(dev_tweets)
        self.tdp.write_predictions(dev_tweet_ids, predictions, "development/multinomial_nb_count_preds.csv")

        print("Start MultinomialNB on tfidf data...")
        train_labels, train_tweet_ids, train_tweets = self.tdp.read_count_tfidf_data("data/train_tfidf.csv")
        dev_labels, dev_tweet_ids, dev_tweets = self.tdp.read_count_tfidf_data("data/dev_tfidf.csv")

        clf = MultinomialNB()
        clf.fit(train_tweets, train_labels)
        predictions = clf.predict(dev_tweets)
        self.tdp.write_predictions(dev_tweet_ids, predictions, "development/multinomial_nb_tfidf_preds.csv")

        print("Start MultinomialNB on glove data...")
        train_labels, train_tweet_ids, train_tweets = self.tdp.read_glove_data("data/train_glove.csv")
        dev_labels, dev_tweet_ids, dev_tweets = self.tdp.read_glove_data("data/dev_glove.csv")

        scaler = MinMaxScaler().fit(train_tweets)
        train_tweets = scaler.transform(train_tweets)
        dev_tweets = scaler.transform(dev_tweets)

        clf = MultinomialNB()
        clf.fit(train_tweets, train_labels)
        predictions = clf.predict(dev_tweets)
        self.tdp.write_predictions(dev_tweet_ids, predictions, "development/multinomial_nb_glove_preds.csv")

    def bernoulli_nb_predictions(self):
        print("Start BernoulliNB on count data...")
        train_labels, train_tweet_ids, train_tweets = self.tdp.read_count_tfidf_data("data/train_count.csv")
        dev_labels, dev_tweet_ids, dev_tweets = self.tdp.read_count_tfidf_data("data/dev_count.csv")

        clf = BernoulliNB()
        clf.fit(train_tweets, train_labels)
        predictions = clf.predict(dev_tweets)
        self.tdp.write_predictions(dev_tweet_ids, predictions, "development/bernoulli_nb_count_preds.csv")

        print("Start BernoulliNB on tfidf data...")
        train_labels, train_tweet_ids, train_tweets = self.tdp.read_count_tfidf_data("data/train_tfidf.csv")
        dev_labels, dev_tweet_ids, dev_tweets = self.tdp.read_count_tfidf_data("data/dev_tfidf.csv")

        clf = BernoulliNB()
        clf.fit(train_tweets, train_labels)
        predictions = clf.predict(dev_tweets)
        self.tdp.write_predictions(dev_tweet_ids, predictions, "development/bernoulli_nb_tfidf_preds.csv")

        print("Start BernoulliNB on glove data...")
        train_labels, train_tweet_ids, train_tweets = self.tdp.read_glove_data("data/train_glove.csv")
        dev_labels, dev_tweet_ids, dev_tweets = self.tdp.read_glove_data("data/dev_glove.csv")

        clf = BernoulliNB()
        clf.fit(train_tweets, train_labels)
        predictions = clf.predict(dev_tweets)
        self.tdp.write_predictions(dev_tweet_ids, predictions, "development/bernoulli_nb_glove_preds.csv")

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

    def zero_r_predictions(self):
        # Count
        predictions_count = self.zero_r(self.train_tweets_count, self.train_labels_count, self.dev_tweets_count)
        self.tdp.write_predictions(self.dev_tweet_ids_count, predictions_count, "development/zero_r_count_preds.csv")

        # TF-IDF
        predictions_tfidf = self.zero_r(self.train_tweets_tfidf, self.train_labels_tfidf, self.dev_tweets_tfidf)
        self.tdp.write_predictions(self.dev_tweet_ids_tfidf, predictions_tfidf, "development/zero_r_tfidf_preds.csv")

        # Glove
        predictions_glove = self.zero_r(self.train_tweets_glove, self.train_labels_glove, self.dev_tweets_glove)
        self.tdp.write_predictions(self.dev_tweet_ids_glove, predictions_glove, "development/zero_r_glove_preds.csv")

    def decision_tree(self, max_depth, train_tweet, train_labels, test_tweet):
        print("Start Decision Tree...")
        dtc = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
        dtc.fit(train_tweet, train_labels)
        predictions = dtc.predict(test_tweet)
        return predictions

    def neural_network(self):
        clf = MLPClassifier(hidden_layer_sizes = (256, 256), max_iter=500)
        print("Start Neural Network on count data...")
        clf.fit(self.train_tweets_count, self.train_labels_count)
        predictions = clf.predict(self.dev_tweets_count)
        self.tdp.write_predictions(self.dev_tweet_ids_count, predictions, "development/nn_256_count_preds.csv")

        print("Start Neural Network on tfidf data...")
        clf.fit(self.train_tweets_tfidf, self.train_labels_tfidf)
        predictions = clf.predict(self.dev_tweets_tfidf)
        self.tdp.write_predictions(self.dev_tweet_ids_tfidf, predictions, "development/nn_256_tfidf_preds.csv")

        print("Start Neural Network on glove data...")
        clf.fit(self.train_tweets_glove, self.train_labels_glove)
        predictions = clf.predict(self.dev_tweets_glove)
        self.tdp.write_predictions(self.dev_tweet_ids_glove, predictions, "development/nn_256_glove_preds.csv")

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

    def k_neighbors(self, neighbours, train_tweet, train_labels, test_tweet):
        print("Start KNN, K = " + str(neighbours) + " ...")
        knc = KNeighborsClassifier(n_neighbors=neighbours)
        knc.fit(train_tweet, train_labels)
        predictions = knc.predict(test_tweet)

        return predictions

    def zero_r(self, train_tweet, train_labels, test_tweet):
        from sklearn.dummy import DummyClassifier
        dc = DummyClassifier(strategy="most_frequent")
        dc.fit(train_tweet, train_labels)
        predictions = dc.predict(test_tweet)
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
    sp.neural_network()


if __name__ == "__main__":
    main()
