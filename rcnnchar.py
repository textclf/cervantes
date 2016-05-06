import csv
import os
import numpy as np

from cervantes.language import TwoLevelsEmbedding
from cervantes.nn.models import LanguageClassifier, RCNNClassifier
from cervantes.box import EnglishCharBox

from keras.models import Model, Sequential

from keras.layers import Input, Reshape, \
    Embedding, GRU, \
    Dense, Highway, \
    Convolution1D,  \
    Dropout, merge, \
    TimeDistributed,\
    Lambda

import keras.backend as K

TRAIN_FILE = "./data/yelp_review_full_csv/train.csv"
TEST_FILE = "./data/yelp_review_full_csv/test.csv"

LOG_FILE = "logs/rcnn-try1.txt"
MODEL_SPEC_FILE = "./rcnn-try1.json"
MODEL_WEIGHTS_FILE = "./rcnn-try1.h5"

EMBEDDING_FILE = "Yelp_stars_model.pkl"

CHARACTERS_PER_WORD = 15
WORDS_PER_DOCUMENT = 300


def parse_file(filepath):
    with open(filepath, "r") as f:
        csv_reader = csv.reader(f)
        labels = []
        texts = []
        for row in csv_reader:
            labels.append(int(row[0]) - 1)
            texts.append(row[1])
        return (texts, labels)


def shuffle_data(train_values, labels):
    combined_lists = zip(train_values, labels)
    np.random.shuffle(combined_lists)
    return zip(*combined_lists)

print "Getting data in format texts / labels"
(train_texts, train_labels) = shuffle_data(*parse_file(TRAIN_FILE))
(test_texts, test_labels) = shuffle_data(*parse_file(TEST_FILE))

# (train_texts, train_labels) = (train_texts[:1000], train_labels[:1000])
# (test_texts, test_labels) = (test_texts[:1000], test_labels[:1000])


print "Building language embedding"
if not os.path.isfile(EMBEDDING_FILE):

    print "Building character embedding"
    cbox = EnglishCharBox(vector_dim=50)

    # Build the language embedding with the given vector box and 2000 chars
    # per text
    lembedding = TwoLevelsEmbedding(
        vector_box=cbox,
        type=TwoLevelsEmbedding.CHAR_WORD_EMBEDDING,
        size_level1=CHARACTERS_PER_WORD,
        size_level2=WORDS_PER_DOCUMENT
    )

    lembedding.compute(train_texts + test_texts)
    lembedding.set_labels(train_labels + test_labels)

    print "Saving embedding"
    lembedding.save(EMBEDDING_FILE)
else:
    print "Embedding already created, loading"
    lembedding = TwoLevelsEmbedding.load(EMBEDDING_FILE)


rcnn = RCNNClassifier(lembedding, num_classes=5, optimizer='adam')

rcnn.train(
    X=lembedding.data[:len(train_labels)],
    y=lembedding.labels[:len(train_labels)],
    model_weights_file=MODEL_WEIGHTS_FILE,
    model_spec_file=MODEL_SPEC_FILE)

rcnn.test(X=lembedding.data[len(train_labels):],
          y=lembedding.labels[len(train_labels):])

rcnn.log_results(LOG_FILE,
                 X_test=lembedding.data[len(train_labels):],
                 y_test=lembedding.labels[len(train_labels):])
