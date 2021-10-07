from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.dummy import DummyClassifier

from dataProcessing import TwitterDataProcessing


class SentimentPrediction:
    def __init__(self):
        self.tdp = TwitterDataProcessing()

        self.train_labels_count, self.train_tweet_ids_count, self.train_tweets_count = self.tdp.read_count_tfidf_data("data/train_count.csv")
        self.dev_labels_count, self.dev_tweet_ids_count, self.dev_tweets_count = self.tdp.read_count_tfidf_data("data/dev_count.csv")

        self.train_labels_tfidf, self.train_tweet_ids_tfidf, self.train_tweets_tfidf = self.tdp.read_count_tfidf_data("data/train_tfidf.csv")
        self.dev_labels_tfidf, self.dev_tweet_ids_tfidf, self.dev_tweets_tfidf = self.tdp.read_count_tfidf_data("data/dev_tfidf.csv")

        self.train_labels_glove, self.train_tweet_ids_glove, self.train_tweets_glove = self.tdp.read_glove_data("data/train_glove.csv")
        self.dev_labels_glove, self.dev_tweet_ids_glove, self.dev_tweets_glove = self.tdp.read_glove_data("data/dev_glove.csv")

    def multinomial_nb(self):
        clf = MultinomialNB()
        print("Start MultinomialNB on count data...")
        clf.fit(self.train_tweets_count, self.train_labels_count)
        predictions = clf.predict(self.dev_tweets_count)
        self.tdp.write_predictions(self.dev_tweet_ids_count, predictions, "development/multinomial_nb_count_preds.csv")

        print("Start MultinomialNB on tfidf data...")
        clf.fit(self.train_tweets_tfidf, self.train_labels_tfidf)
        predictions = clf.predict(self.dev_tweets_tfidf)
        self.tdp.write_predictions(self.dev_tweet_ids_tfidf, predictions, "development/multinomial_nb_tfidf_preds.csv")

        print("Start MultinomialNB on glove data...")
        scaler = MinMaxScaler().fit(self.train_tweets_glove)
        train_tweets_glove = scaler.transform(self.train_tweets_glove)
        dev_tweets_glove = scaler.transform(self.dev_tweets_glove)
        clf.fit(train_tweets_glove, self.train_labels_glove)
        predictions = clf.predict(dev_tweets_glove)
        self.tdp.write_predictions(self.dev_tweet_ids_glove, predictions, "development/multinomial_nb_glove_preds.csv")

    def bernoulli_nb(self):
        clf = BernoulliNB()
        print("Start BernoulliNB on count data...")
        clf.fit(self.train_tweets_count, self.train_labels_count)
        predictions = clf.predict(self.dev_tweets_count)
        self.tdp.write_predictions(self.dev_tweet_ids_count, predictions, "development/bernoulli_nb_count_preds.csv")

        print("Start BernoulliNB on tfidf data...")
        clf.fit(self.train_tweets_tfidf, self.train_labels_tfidf)
        predictions = clf.predict(self.dev_tweets_tfidf)
        self.tdp.write_predictions(self.dev_tweet_ids_tfidf, predictions, "development/bernoulli_nb_tfidf_preds.csv")

        print("Start BernoulliNB on glove data...")
        clf.fit(self.train_tweets_glove, self.train_labels_glove)
        predictions = clf.predict(self.dev_tweets_glove)
        self.tdp.write_predictions(self.dev_tweet_ids_glove, predictions, "development/bernoulli_nb_glove_preds.csv")

    def neural_network(self, hidden_layer):
        clf = MLPClassifier(hidden_layer_sizes = hidden_layer, activation="logistic", early_stopping=True, max_iter=500, batch_size=3)
        print("Start Neural Network on count data...")
        clf.fit(self.train_tweets_count, self.train_labels_count)
        predictions = clf.predict(self.dev_tweets_count)
        self.tdp.write_predictions(self.dev_tweet_ids_count, predictions, "development/nn_log_adam_b3_" + str(hidden_layer) + "_count_preds.csv")

        print("Start Neural Network on tfidf data...")
        clf.fit(self.train_tweets_tfidf, self.train_labels_tfidf)
        predictions = clf.predict(self.dev_tweets_tfidf)
        self.tdp.write_predictions(self.dev_tweet_ids_tfidf, predictions, "development/nn_log_adam_b3_" + str(hidden_layer) + "_tfidf_preds.csv")

        print("Start Neural Network on glove data...")
        clf.fit(self.train_tweets_glove, self.train_labels_glove)
        predictions = clf.predict(self.dev_tweets_glove)
        self.tdp.write_predictions(self.dev_tweet_ids_glove, predictions, "development/nn_log_adam_b3_" + str(hidden_layer) + "_glove_preds.csv")

    def logistic_regression(self):
        clf = LogisticRegression(max_iter=500)
        print("Start Logistic Regression on count data...")
        clf.fit(self.train_tweets_count, self.train_labels_count)
        predictions = clf.predict(self.dev_tweets_count)
        self.tdp.write_predictions(self.dev_tweet_ids_count, predictions, "development/lr_count_preds.csv")

        print("Start Logistic Regression on tfidf data...")
        clf.fit(self.train_tweets_tfidf, self.train_labels_tfidf)
        predictions = clf.predict(self.dev_tweets_tfidf)
        self.tdp.write_predictions(self.dev_tweet_ids_tfidf, predictions, "development/lr_tfidf_preds.csv")

        print("Start Logistic Regression on glove data...")
        clf.fit(self.train_tweets_glove, self.train_labels_glove)
        predictions = clf.predict(self.dev_tweets_glove)
        self.tdp.write_predictions(self.dev_tweet_ids_glove, predictions, "development/lr_glove_preds.csv")

    def k_nearest_neighbor(self, neighbour):
        clf = KNeighborsClassifier(n_neighbors=neighbour)
        print("Start K Nearest Neighbour on count data...")
        clf.fit(self.train_tweets_count, self.train_labels_count)
        predictions = clf.predict(self.dev_tweets_count)
        self.tdp.write_predictions(self.dev_tweet_ids_count, predictions, "development/knn_" + str(neighbour) + "_count_preds.csv")

        print("Start K Nearest Neighbour on tfidf data...")
        clf.fit(self.train_tweets_tfidf, self.train_labels_tfidf)
        predictions = clf.predict(self.dev_tweets_tfidf)
        self.tdp.write_predictions(self.dev_tweet_ids_tfidf, predictions, "development/knn_" + str(neighbour) + "_tfidf_preds.csv")

        print("Start K Nearest Neighbour on glove data...")
        clf.fit(self.train_tweets_glove, self.train_labels_glove)
        predictions = clf.predict(self.dev_tweets_glove)
        self.tdp.write_predictions(self.dev_tweet_ids_glove, predictions, "development/knn_" + str(neighbour) + "_glove_preds.csv")

    def zero_r(self):
        clf = DummyClassifier(strategy="most_frequent")
        print("Start Zero R on count data...")
        clf.fit(self.train_tweets_count, self.train_labels_count)
        predictions = clf.predict(self.dev_tweets_count)
        self.tdp.write_predictions(self.dev_tweet_ids_count, predictions, "development/zero_r_count_preds.csv")

        print("Start Zero R on tfidf data...")
        clf.fit(self.train_tweets_tfidf, self.train_labels_tfidf)
        predictions = clf.predict(self.dev_tweets_tfidf)
        self.tdp.write_predictions(self.dev_tweet_ids_tfidf, predictions, "development/zero_r_tfidf_preds.csv")

        print("Start Zero R on glove data...")
        clf.fit(self.train_tweets_glove, self.train_labels_glove)
        predictions = clf.predict(self.dev_tweets_glove)
        self.tdp.write_predictions(self.dev_tweet_ids_glove, predictions, "development/zero_r_glove_preds.csv")


def main():
    sp = SentimentPrediction()
    sp.k_nearest_neighbor(1)


if __name__ == "__main__":
    main()
