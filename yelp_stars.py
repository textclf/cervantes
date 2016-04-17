import csv
import os
import numpy as np

from cervantes.box import WordVectorBox
from cervantes.language import OneLevelEmbedding
from cervantes.nn.models import RNNClassifier

TRAIN_FILE = "/home/alfredo/Desktop/yelp_review_full_csv/train.csv"
TEST_FILE = "/home/alfredo/Desktop/yelp_review_full_csv/test.csv"

WV_FILE = '../deep-text/embeddings/wv/glove.42B.300d.120000.txt'

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

print "Building language embeddings. This requires parsing text so it might " \
      "be pretty slow "
# Compute text embeddings, containing the processed text tokens together with a vector-to-index
# translation object (the vector box), should be pickled in order to be efficiently used with
# different models. Hence, we can save time once we have precomputed a language embedding
EMBEDDING_FILE = "Yelp_stars_small.pkl"
if not os.path.isfile(EMBEDDING_FILE):

    # We need a file with precomputed wordvectors
    print 'Building global word vectors from {}'.format(WV_FILE)

    gbox = WordVectorBox(WV_FILE)
    gbox.build(zero_token=True, normalize_variance=False, normalize_norm=True)

    # Build the language embedding with the given vector box and 300 words per text
    lembedding = OneLevelEmbedding(gbox, size=300)
    lembedding.compute(train_texts + test_texts)
    lembedding.save(EMBEDDING_FILE)
else:
    lembedding = OneLevelEmbedding.load(EMBEDDING_FILE)

# Create a recurrent neural network model and train it, the data from the computed
# embedding must be used
gru = RNNClassifier(lembedding, num_classes=5, unit='gru',
                    rnn_size=32, train_vectors=True)
gru.train(X=lembedding.data[:len(train_texts)], y=train_labels,
          model_weights_file="test.weights", model_spec_file="test.spec")
gru.test_sequential(X=lembedding.data[len(train_texts):], y=test_labels, verbose=True)
gru.log_results("log.txt", X_test=lembedding.data[len(train_texts):], y_test=test_labels)


# from classic.classifiers import TextClassifier, NaiveBayesClassifier, SGDTextClassifier, \
#     LogisticClassifier, SVMClassifier, PerceptronClassifier, RandomForestTextClassifier
# print "Naive Bayes"
# nb = NaiveBayesClassifier()
# nb.set_training_data(train_texts, train_labels)
# nb.set_test_data(test_texts, test_labels)
# nb.set_bag_of_ngrams()
#
# nb.train()
# train_error = nb.get_training_error()
# test_error = nb.get_test_error()
# print "Training error: " + str(train_error)
# print "Test error: " + str(test_error)