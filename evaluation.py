import csv
import pandas as pd
from sklearn.metrics import accuracy_score
from dataProcessing import TwitterDataProcessing


class PerformanceEvaluation:
    def __init__(self):
        self.tdp = TwitterDataProcessing()

    def evaluation(self):
        summary_file = open("development/summary.csv", "w")
        writer = csv.writer(summary_file)
        writer.writerow(["Method/Accuracy", "BoW", "TF-IDF", "GloVe"])

        dev_labels, dev_tweet_ids, dev_tweets = self.tdp.read_count_tfidf_data("data/dev_count.csv")
        pred_files = [["Zero R", "zero_r_count", "zero_r_tfidf", "zero_r_glove"],
                      ["KNN (K = 101)", "knn_101_count", "knn_101_tfidf", "knn_101_glove"],
                      ["Multinomial Naive Bayes", "multinomial_nb_count", "multinomial_nb_tfidf", "multinomial_nb_glove"],
                      ["Bernoulli Naive Bayes", "bernoulli_nb_count", "bernoulli_nb_tfidf", "bernoulli_nb_glove"],
                      ["Logistic Regression", "lr_count", "lr_tfidf", "lr_glove"],
                      ["Multilayer Perceptron (hidden_layer = (10, 5))", "mp_logistic_(10, 5)_count", "mp_logistic_(10, 5)_tfidf", "mp_logistic_(10, 5)_glove"],
                      ["Multilayer Perceptron (hidden_layer = (50, 25))", "mp_logistic_(50, 25)_count", "mp_logistic_(50, 25)_tfidf", "mp_logistic_(50, 25)_glove"],
                      ["Multilayer Perceptron (hidden_layer = (100, 50))", "mp_logistic_(100, 50)_count", "mp_logistic_(100, 50)_tfidf", "mp_logistic_(100, 50)_glove"]]

        for pred_file in pred_files:
            pred_count = pd.read_csv("development/" + pred_file[1] + "_preds.csv")["sentiment"]
            acc_score_count = round(accuracy_score(dev_labels, pred_count), 4)

            pred_tfidf = pd.read_csv("development/" + pred_file[2] + "_preds.csv")["sentiment"]
            acc_score_tfidf = round(accuracy_score(dev_labels, pred_tfidf), 4)

            pred_glove = pd.read_csv("development/" + pred_file[3] + "_preds.csv")["sentiment"]
            acc_score_glove = round(accuracy_score(dev_labels, pred_glove), 4)

            writer.writerow([pred_file[0], acc_score_count, acc_score_tfidf, acc_score_glove])


def main():
    pe = PerformanceEvaluation()
    pe.evaluation()


if __name__ == "__main__":
    main()
