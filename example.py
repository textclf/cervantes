import os

from nlpdatahandlers import YelpDataHandler

from box import WordVectorBox
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

    print "Building language embeddings. This requires parsing text so it might" \
          "be pretty slow "
    # Compute text embeddings, containing the processed text tokens together with a vector-to-index
    # translation object (the vector box), should be pickled in order to be efficiently used with
    # different models. Hence, we can save time once we have precomputed a language embedding
    EMBEDDING_FILE = "YelpFunny1Level.pkl"
    if not os.path.isfile(EMBEDDING_FILE):

        # We need a file with precomputed wordvectors
        print 'Building global word vectors from {}'.format(WV_FILE)

        gbox = WordVectorBox(WV_FILE)
        gbox.build(zero_token=True, normalize_variance=False, normalize_norm=True)

        # Build the language embedding with the given vector box and 300 words per text
        lembedding = OneLevelEmbedding(gbox, size=300)
        lembedding.compute(train_reviews + test_reviews)
        lembedding.save(EMBEDDING_FILE)
    else:
        lembedding = OneLevelEmbedding.load(EMBEDDING_FILE)

    # Create a recurrent neural network model and train it, the data from the computed
    # embedding must be used
    gru = RNNBinaryClassifier(lembedding, unit='gru', rnn_size=64, train_vectors=True)
    gru.train(X=lembedding.data[:1000], y=train_labels[:1000])
    gru.test(X=lembedding.data[1001:2000], y=train_labels[1001:2000])
    gru.log_results()

    # We can also specify a custom Keras model expecting indexed texts as seen in the embeddings
    model = Sequential()
    # # TODO: Provide a generic builder of initial layers for embeddings
    model.add(Embedding(gbox.W.shape[0], gbox.W.shape[1], weights=[gbox.W],
                        input_length=train_reviews.shape[1]))
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(GRU(64, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(GRU(16, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('tanh'))

    model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

    my_clf = LanguageClassifier(model, type=LanguageClassifier.SEQUENTIAL)
    my_clf.train(X=lembedding.data[:1000], y=train_labels[:1000])
    my_clf.test(X=lembedding.data[1001:2000], y=train_labels[1001:2000])
    my_clf.log_results()
