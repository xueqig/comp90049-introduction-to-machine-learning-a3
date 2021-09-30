import csv
import pandas as pd
from sklearn.metrics import accuracy_score
from DataProcessing import TwitterDataProcessing


class PerformanceEvaluation:
    def __init__(self):
        self.tdp = TwitterDataProcessing()

    def evaluation(self):
        summary_file = open("development/summary.csv", "w")
        writer = csv.writer(summary_file)
        writer.writerow(["Method", "Accuracy Score"])

        dev_labels, dev_tweet_ids, dev_tweets = self.tdp.read_count_tfidf_data("data/dev_count.csv")
        pred_files = ["knn_1_count", "knn_3_count", "knn_5_count", "knn_7_count",
                      "knn_1_tfidf", "knn_3_tfidf", "knn_5_tfidf", "knn_7_tfidf",
                      "knn_1_glove", "knn_3_glove", "knn_5_glove", "knn_7_glove",
                      "nb_count", "nb_tfidf", "nb_glove",
                      "lr_count", "lr_tfidf", "lr_glove",
                      "nn_count", "nn_tfidf", "nn_glove",
                      "dt_count", "dt_tfidf", "dt_glove"]
        for pred_file in pred_files:
            predictions = pd.read_csv("development/" + pred_file + "_preds.csv")["sentiment"]
            acc_score = accuracy_score(dev_labels, predictions)
            writer.writerow([pred_file, acc_score])

def main():
    pe = PerformanceEvaluation()
    pe.evaluation()


if __name__ == "__main__":
    main()