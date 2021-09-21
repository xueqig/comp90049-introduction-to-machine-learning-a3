import csv
import math
from random import random


class TwitterData:
    def __init__(self):
        self.train_labels = []
        self.train_tweet_ids = []
        self.test_tweet_ids = []
        self.read_train_full()
        self.read_test_full()
        self.write_predictions(self.weight_rand_preds(), "weight_rand_preds.csv")
        self.write_predictions(self.rand_preds(), "rand_preds.csv")

    def rand_preds(self):
        labels = ["pos", "neu", "neg"]
        rand_preds = []
        for i in range(len(self.test_tweet_ids)):
            rand_idx = math.floor(len(labels) * random())
            rand_preds.append(labels[rand_idx])
        return rand_preds

    def weight_rand_preds(self):
        train_len = len(self.train_tweet_ids)
        train_pos_pct = self.train_labels.count("pos") / train_len
        train_neu_pct = self.train_labels.count("neu") / train_len
        train_neg_pct = self.train_labels.count("neg") / train_len

        test_len = len(self.test_tweet_ids)
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

    def read_train_full(self):
        with open("data/train_full.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    self.train_labels.append(row[0])
                    self.train_tweet_ids.append(row[1])
                    line_count += 1

    def read_test_full(self):
        with open("data/test_full.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    self.test_tweet_ids.append(row[1])
                    line_count += 1

    def get_train_labels(self):
        return self.train_labels

    def write_predictions(self, predictions, filename):
        print("Writing " + filename + "...")
        pred_file = open(filename, "w")
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
