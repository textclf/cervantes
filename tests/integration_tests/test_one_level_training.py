import os

from cervantes.box import WordVectorBox
from cervantes.language import OneLevelEmbedding
from cervantes.nn.models import RNNClassifier
from tests.common import *

from keras.callbacks import EarlyStopping

MODEL_WEIGHTS_FILE = "remove.weights"
MODEL_SPEC_FILE = "remove.spec"
LOG_FILE = "remove.log"
EMBEDDING_FILE = "embedding.plk"

def test_simple_rnn_train():

    gbox = WordVectorBox(WORD_VECTORS_FILE, verbose=False)
    train_texts, train_labels, test_texts, test_labels = get_yelp_polarity_data()
    train_texts, train_labels = train_texts[:10000], train_labels[:10000]
    test_texts, test_labels = test_texts[:500], test_labels[:500]

    gbox.build(zero_token=True, normalize_variance=False, normalize_norm=True)

    print "Computing embedding"
    lembedding = OneLevelEmbedding(gbox, size=180)

    lembedding.compute_word_repr(train_texts + test_texts)
    lembedding.set_labels(train_labels + test_labels)
    lembedding.save(EMBEDDING_FILE)

    clf = RNNClassifier(lembedding, num_classes=2, unit='gru',
                        rnn_size=16, train_vectors=True)

    FIT_PARAMS = {
        "batch_size": 32,
        "nb_epoch": 3,
        "verbose": 2,
        "validation_split": 0.15,
        "callbacks": [EarlyStopping(verbose=True, patience=5, monitor='val_acc')]
    }

    clf.train(X=lembedding.data[:len(train_labels)], y=lembedding.labels[:len(train_labels)],
              model_weights_file=MODEL_WEIGHTS_FILE, model_spec_file=MODEL_SPEC_FILE,
              **FIT_PARAMS)
    (acc, prec, recall) = clf.test(X=lembedding.data[len(train_labels):],
                                   y=lembedding.labels[len(train_labels):])
    clf.log_results(LOG_FILE, X_test=lembedding.data[len(train_labels):],
                    y_test=lembedding.labels[len(train_labels):])

    assert os.path.isfile(MODEL_WEIGHTS_FILE)
    assert os.path.isfile(MODEL_SPEC_FILE)
    assert os.path.isfile(LOG_FILE)
    assert os.path.isfile(EMBEDDING_FILE)

    assert acc > 0.65

    os.remove(MODEL_WEIGHTS_FILE)
    os.remove(MODEL_SPEC_FILE)
    os.remove(LOG_FILE)
    os.remove(EMBEDDING_FILE)