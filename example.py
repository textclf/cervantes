import os

from nlpdatahandlers import YelpDataHandler, ImdbDataHandler
from cervantes.box import WordVectorBox
from cervantes.language import OneLevelEmbedding
from cervantes.nn.models import RNNClassifier

YELP_FUNNY_TRAIN = '../yelp-dataset/TrainSet_funny_75064'
YELP_FUNNY_DEV = '../yelp-dataset/DevSet_funny_75064'
YELP_FUNNY_TEST = '../yelp-dataset/TestSet_funny_75064'

IMDB_DATA = '../deep-text/datasets/aclImdb/aclImdb'
WV_FILE = '../deep-text/embeddings/wv/glove.42B.300d.120000.txt'

if __name__ == '__main__':

    print "Getting data in format texts / labels"

    imdb = ImdbDataHandler(source=IMDB_DATA)
    (train_reviews, train_labels) = imdb.get_data(type=ImdbDataHandler.DATA_TRAIN)
    (test_reviews, test_labels) = imdb.get_data(type=ImdbDataHandler.DATA_TEST)
    train_reviews = train_reviews[:5000]
    test_reviews = test_reviews[:1000]
    train_labels = list(train_labels)[:5000]
    test_labels = list(test_labels)[:1000]

    #yelp = YelpDataHandler()
    #(train_reviews, train_labels, test_reviews, test_labels) = \
    #    yelp.get_data(YELP_FUNNY_TRAIN, YELP_FUNNY_DEV, YELP_FUNNY_TEST)

    print "Building language embeddings. This requires parsing text so it might " \
          "be pretty slow "
    # Compute text embeddings, containing the processed text tokens together with a vector-to-index
    # translation object (the vector box), should be pickled in order to be efficiently used with
    # different models. Hence, we can save time once we have precomputed a language embedding
    EMBEDDING_FILE = "Imdb_example2.pkl"
    if not os.path.isfile(EMBEDDING_FILE):

        # We need a file with precomputed wordvectors
        print 'Building global word vectors from {}'.format(WV_FILE)

        gbox = WordVectorBox(WV_FILE)
        gbox.build(zero_token=True, normalize_variance=False, normalize_norm=True)

        # Build the language embedding with the given vector box and 300 words per text
        lembedding = OneLevelEmbedding(gbox, size=300)
        lembedding.compute(train_reviews + test_reviews)
        lembedding.set_labels(train_labels + test_labels)
        lembedding.save(EMBEDDING_FILE)
    else:
        lembedding = OneLevelEmbedding.load(EMBEDDING_FILE)

    # Create a recurrent neural network model and train it, the data from the computed
    # embedding must be used
    gru = RNNClassifier(lembedding, num_classes=2, unit='gru',
                        rnn_size=16, train_vectors=False)
    gru.train(X=lembedding.data[:len(train_labels)], y=lembedding.labels[:len(train_labels)],
              model_weights_file="test.weights", model_spec_file="test.spec")
    gru.test_sequential(X=lembedding.data[len(train_labels):],
                        y=lembedding.labels[len(train_labels):])
    gru.log_results("log.txt", X_test=lembedding.data[len(train_labels):],
                    y_test=lembedding.labels[len(train_labels):])