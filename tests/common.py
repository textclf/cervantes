"""
Data and functions necessary for the tests
"""
import csv
import numpy as np

WORD_VECTORS_FILE = "../../deep-text/embeddings/wv/glove.42B.300d.120000.txt"
AG_NEWS_TRAIN = "../../ag_news_csv/train.csv"
AG_NEWS_TEST = "../../ag_news_csv/test.csv"
YELP_POLARITY_TRAIN = "../../yelp_review_polarity_csv/train.csv"
YELP_POLARITY_TEST = "../../yelp_review_polarity_csv/test.csv"

def shuffle_data(train_values, labels):
        combined_lists = zip(train_values, labels)
        np.random.shuffle(combined_lists)
        zipped = zip(*combined_lists)
        return list(zipped[0]), list(zipped[1])

def clean(texts):
    cleaned_texts = []
    for text in texts:
        new_text = text.replace("\\n", "\n")
        new_text = new_text.replace('\\"', '"')
        cleaned_texts.append(new_text)
    return cleaned_texts

def parse_ag_news(filepath):
    with open(filepath, "r") as f:
        csv_reader = csv.reader(f)
        labels = []
        texts = []
        for row in csv_reader:
            labels.append(int(row[0]) - 1)
            texts.append(row[1] + ".  " + row[2])
        texts = clean(texts)
        return texts, labels

def get_ag_news_data():
    (train_texts, train_labels) = shuffle_data(*parse_ag_news(AG_NEWS_TRAIN))
    (test_texts, test_labels) = shuffle_data(*parse_ag_news(AG_NEWS_TEST))

    return train_texts, train_labels, test_texts, test_labels

def parse_yelp_polarity(filepath):
    with open(filepath, "r") as f:
        csv_reader = csv.reader(f)
        labels = []
        texts = []
        for row in csv_reader:
            labels.append(int(row[0]) - 1)
            texts.append(row[1])
        texts = clean(texts)
        return texts, labels

def get_yelp_polarity_data():
    (train_texts, train_labels) = shuffle_data(*parse_yelp_polarity(YELP_POLARITY_TRAIN))
    (test_texts, test_labels) = shuffle_data(*parse_yelp_polarity(YELP_POLARITY_TEST))

    return train_texts, train_labels, test_texts, test_labels
