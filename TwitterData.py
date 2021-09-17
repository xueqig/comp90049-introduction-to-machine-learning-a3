import csv


class TwitterData:
    def __init__(self):
        self.train_labels = []
        self.read_csv()

    def read_csv(self):
        with open("data/train_full.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    self.train_labels.append(row[0])
                    line_count += 1

    def get_train_labels(self):
        return self.train_labels


def main():
    td = TwitterData()
    print(td.get_train_labels()[:5])


if __name__ == "__main__":
    main()
