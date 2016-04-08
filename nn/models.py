from keras.layers.recurrent import GRU, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers.core import Dense, Activation, Dropout, Reshape, Flatten

from language import OneLevelEmbedding, TwoLevelsEmbedding

class LanguageClassifierException(Exception):
    pass

class LanguageClassifier(object):

    SEQUENTIAL = 0
    FUNCTIONAL = 1

    def __init__(self, model, type=SEQUENTIAL):
        self.model = model
        self.type = type

    def train(self, X, y, train_options):
        if self.type == LanguageClassifier.SEQUENTIAL:
            self.train_sequential(X, y, "TODO", fit_params=train_options)

    def train_sequential(self, X, y, model_file, fit_params=None, monitor='val_acc'):
        # TODO: DOCUMENT once thoroughly tested

        if fit_params is None:
            fit_params = {
                "batch_size": 32,
                "nb_epoch": 45,
                "verbose": True,
                "validation_split": 0.15,
                "show_accuracy": True,
                "callbacks": [EarlyStopping(verbose=True, patience=5, monitor=monitor),
                              ModelCheckpoint(model_file, monitor=monitor, verbose=True, save_best_only=True)]
            }
        print 'Fitting! Hit CTRL-C to stop early...'

        try:
            history = self.model.fit(X, y, **fit_params)
        except KeyboardInterrupt:
            print "Training stopped early!"
            history = self.model.history

        return history

    def test_sequential(self, X_test, y_test, saved_model=None):
        # TODO: DOCUMENT

        if saved_model is not None:
            self.model.load_weights(saved_model)

        print "Getting predictions on the test set"
        yhat = self.model.predict(X_test, verbose=True, batch_size=50)

        # TODO: Obtain new metrics from Keras
        acc = ((yhat.ravel() > 0.5) == (y_test > 0.5)).mean()

        print "Test set accuracy of {}%.".format(acc * 100.0)
        print "Test set error of {}%. Exiting...".format((1 - acc) * 100.0)

        return acc

    def log_results(self):
        pass

class RNNBinaryClassifier(LanguageClassifier):

    # TODO: Maybe we can pass directly a vector box rather than the embedding since
    # only the first W from vector box is used
    def __init__(self, lembedding, unit='gru', rnn_size=128, train_vectors=True):

        if not isinstance(lembedding, OneLevelEmbedding):
            raise LanguageClassifierException("The model only accepts one-level language embeddings")

        model = self._generate_model(lembedding, unit, rnn_size, train_vectors)
        super(RNNBinaryClassifier, self).__init__(model, LanguageClassifier.SEQUENTIAL)

    @staticmethod
    def _generate_model(lembedding, unit='gru', rnn_size=128, train_vectors=True):

        model = Sequential()
        # TODO: add sizes to vector boxes / language embedding classes
        emb = Embedding(lembedding.vector_box.W.shape[0],
                        lembedding.vector_box.W.shape[1],
                        weights=lembedding.vector_box.W, W_constraint=None)
        emb.trainable = train_vectors
        model.add(emb)
        if unit == 'gru':
            model.add(GRU(rnn_size, init='uniform', ))
        else:
            model.add(LSTM(rnn_size, init='uniform', ))
        model.add(Dropout(0.2))
        model.add(Dense(1, init='uniform'))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

        return model

class RCNNBinaryClassifier(LanguageClassifier):

    def __init__(self, lembedding, etc):
        if not isinstance(lembedding, TwoLevelsEmbedding):
            raise LanguageClassifierException("The model only accepts two-levels language embeddings")