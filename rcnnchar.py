import csv
import os
import numpy as np

from cervantes.language import TwoLevelsEmbedding
from cervantes.nn.models import LanguageClassifier
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

EMBEDDING_FILE = "Yelp_stars_char.pkl"

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


print "Building language embedding"
if not os.path.isfile(EMBEDDING_FILE):

    print "Building character embedding"
    cbox = EnglishCharBox(vector_dim=100)
    NUMBER_CHARACTERS = cbox.n_chars + 1

    # Build the language embedding with the given vector box and 2000 chars per text
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
    NUMBER_CHARACTERS = lembedding.vector_box.n_chars + 1



print 'Building model...'
NGRAMS = [1, 2, 3, 4, 5]
NFILTERS = 32 * 3

EMBEDDING_DIM = 100
INPUT_SHAPE = (CHARACTERS_PER_WORD * WORDS_PER_DOCUMENT, )
EMBEDDING_SHAPE = (WORDS_PER_DOCUMENT, CHARACTERS_PER_WORD, EMBEDDING_DIM)

doc = Input(shape=(INPUT_SHAPE[0], ), dtype='int32')

embedded = Sequential([
        Embedding(
            input_dim=NUMBER_CHARACTERS, 
            output_dim=EMBEDDING_DIM, 
            input_length=INPUT_SHAPE[0]
            ), 
        Reshape(EMBEDDING_SHAPE)
    ])(doc)

def sub_model(n):
    return Sequential([
            Convolution1D(NFILTERS, n, 
                activation='relu', 
                input_shape=EMBEDDING_SHAPE[1:]
                ), 
            Lambda(
                lambda x: K.max(x, axis=1), 
                output_shape=(NFILTERS,)
                )
        ])


rep = Dropout(0.5)(
    merge(
        [TimeDistributed(sub_model(n))(embedded) for n in NGRAMS], 
        mode='concat', 
        concat_axis=-1
    )
)


output = Dropout(0.5)(
    merge(
        [GRU(90)(rep), GRU(90, go_backwards=True)(rep)], 
        mode='concat', 
        concat_axis=-1
    )
)

mapping = [
        Highway(activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(5, activation='softmax')
    ]

for f in mapping:
    output = f(output)

nn = Model(input=doc, output=output)

nn.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

rcnn = LanguageClassifier(nn, optimizer='adam')

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





