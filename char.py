import os

from nlpdatahandlers import YelpDataHandler

from box import EnglishCharBox
from language.embeddings import OneLevelEmbedding, TwoLevelsEmbedding
from nn.models import RNNBinaryClassifier, LanguageClassifier

from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers.core import Dense, Activation, Dropout, Reshape, Flatten

YELP_FUNNY_TRAIN = '../yelp-dataset/TrainSet_funny_75064'
YELP_FUNNY_DEV = '../yelp-dataset/DevSet_funny_75064'
YELP_FUNNY_TEST = '../yelp-dataset/TestSet_funny_75064'

WV_FILE = '../deep-text/embeddings/wv/glove.42B.300d.120000.txt'

if __name__ == '__main__':

    print "Getting data in format texts / labels"
    yelp = YelpDataHandler()
    (train_reviews, train_labels, test_reviews, test_labels) = \
        yelp.get_data(YELP_FUNNY_TRAIN, YELP_FUNNY_DEV, YELP_FUNNY_TEST)

    print "Building character embedding"
    EMBEDDING_FILE = "YelpChar.pkl"
    if not os.path.isfile(EMBEDDING_FILE):

        cbox = EnglishCharBox(vector_dim=300)

        # Build the language embedding with the given vector box and 300 words per text
        lembedding = OneLevelEmbedding(cbox, type=OneLevelEmbedding.CHAR_EMBEDDING, size=5000)
        lembedding.compute(train_reviews)
        lembedding.save(EMBEDDING_FILE)
    else:
        lembedding = OneLevelEmbedding.load(EMBEDDING_FILE)

    # Create a recurrent neural network model and train it, the data from the computed
    # embedding must be used
    gru = RNNBinaryClassifier(lembedding, unit='gru', rnn_size=64, train_vectors=True)
    gru.train(X=lembedding.data[:1000], y=train_labels[:1000])
    gru.test(X=lembedding.data[1001:2000], y=train_labels[1001:2000])
    #gru.log_results()