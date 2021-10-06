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
        pred_files = ["knn_101_count", "multinomial_nb_count", "bernoulli_nb_count", "lr_count",
                      "nn_log_adam_3_count", "nn_log_adam_64_count", "nn_sgd_3_count", "nn_log_sgd_3_count",
                      "nn_sgd_(3, 1)_count", "nn_log_adam_(3, 3)_count", "nn_log_adam_(64, 64)_count",
                      "nn_log_adam_(256, 128, 64)_count", "nn_log_adam_(256, 128, 64, 32)_count",
                      "nn_log_adam_(512, 256, 128)_count", "nn_log_adam_(200, 100, 50)_count",
                      "nn_sgd_(3, 3)_count", "nn_log_sgd_(3, 3)_count",

                      "dt_count", "zero_r_count",
                      "knn_101_tfidf", "multinomial_nb_tfidf", "bernoulli_nb_tfidf", "lr_tfidf",
                      "nn_log_adam_3_tfidf", "nn_log_adam_64_tfidf", "nn_sgd_3_tfidf", "nn_log_sgd_3_tfidf",
                      "nn_sgd_(3, 1)_tfidf", "nn_log_adam_(3, 3)_tfidf", "nn_log_adam_(64, 64)_tfidf",
                      "nn_log_adam_(256, 128, 64)_tfidf", "nn_log_adam_(256, 128, 64, 32)_tfidf",
                      "nn_log_adam_(512, 256, 128)_tfidf", "nn_log_adam_(200, 100, 50)_tfidf",
                      "nn_sgd_(3, 3)_tfidf", "nn_log_sgd_(3, 3)_tfidf",
                      "dt_tfidf", "zero_r_tfidf",

                      "knn_101_glove", "multinomial_nb_glove", "bernoulli_nb_glove", "lr_glove",
                      "nn_log_adam_3_glove", "nn_log_adam_64_glove", "nn_sgd_3_glove", "nn_log_sgd_3_glove",
                      "nn_sgd_(3, 1)_glove", "nn_log_adam_(3, 3)_glove", "nn_log_adam_(64, 64)_glove",
                      "nn_log_adam_(256, 128, 64)_glove", "nn_log_adam_(256, 128, 64, 32)_glove",
                      "nn_log_adam_(512, 256, 128)_glove", "nn_log_adam_(200, 100, 50)_glove",
                      "nn_sgd_(3, 3)_glove", "nn_log_sgd_(3, 3)_glove",
                      "dt_glove", "zero_r_glove"]
        for pred_file in pred_files:
            predictions = pd.read_csv("development/" + pred_file + "_preds.csv")["sentiment"]
            acc_score = accuracy_score(dev_labels, predictions)
            writer.writerow([pred_file, acc_score])

def main():
    pe = PerformanceEvaluation()
    pe.evaluation()


if __name__ == "__main__":
    main()
