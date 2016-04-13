from __future__ import print_function
from keras.layers.recurrent import GRU, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers.core import Dense, Activation, Dropout, Reshape, Flatten
import keras.utils.np_utils
import keras.models
from time import strftime

from language import OneLevelEmbedding, TwoLevelsEmbedding

class LanguageClassifierException(Exception):
    pass

class LanguageClassifier(object):

    SEQUENTIAL = 0
    FUNCTIONAL = 1

    def __init__(self, model, type=SEQUENTIAL):
        self.model = model
        self.type = type
        self.binary = "unk"  # unknown until training

        self.ttime_start = None
        self.ttime_stop = None  # for training logging purposes

    def save_model(self, model_spec_file, model_weights_file):
        with open(model_spec_file, "w") as f:
            f.write(self.model.to_json())
            f.write("\n")
            if self.binary == "unk":
                f.write("Binary: unk" + "\n")
            else:
                f.write("Binary: " + str(self.binary) + "\n")

            if self.type == self.SEQUENTIAL:
                f.write("Type: sequential\n")
            else:
                f.write("Type: functional\n")

        self.model.save_weights(model_weights_file, overwrite=True)

    @staticmethod
    def load_model(model_spec_file, model_weights_file):
        with open(model_spec_file, "r") as f:
            model = keras.models.model_from_json(f.readline())
            binary = f.readline().strip().split(" ")[1]
            if binary == "True":
                binary = True
            elif binary == "False":
                binary = False

            type = f.readline().strip().split(" ")[1]
            if type == "sequential":
                type = LanguageClassifier.SEQUENTIAL
            else:
                type = LanguageClassifier.FUNCTIONAL

        model.load_weights(model_weights_file)
        lc = LanguageClassifier(model, type)
        lc.binary = binary
        return lc

    def train(self, X, y, model_file=None, train_options=None):
        if self.type == LanguageClassifier.SEQUENTIAL:
            self.train_sequential(X, y, model_file, train_options)
        else:
            self.train_functional(X, y, model_file, train_options)

    def train_sequential(self, X, y, model_file=None, fit_params=None):

        # Provide some default values as training options
        if fit_params is None:
            fit_params = {
                "batch_size": 32,
                "nb_epoch": 2,
                "verbose": 1,
                "validation_split": 0.15,
                "callbacks": [EarlyStopping(verbose=True, patience=5, monitor='val_acc')]
            }
            if model_file is not None:
                fit_params["callbacks"].append(ModelCheckpoint(model_file, monitor='val_acc', verbose=True, save_best_only=True))

        if max(y) > 1:
            Y = keras.utils.np_utils.to_categorical(y)
            self.binary = False
        else:
            Y = y
            self.binary = True

        try:
            print('Fitting! Hit CTRL-C to stop early...')
            self.ttime_start = strftime("%Y-%m-%d %H:%M:%S")
            self.model.fit(X, Y, **fit_params)
        except KeyboardInterrupt:
            print("Training stopped early!")
        self.ttime_stop = strftime("%Y-%m-%d %H:%M:%S")

    def train_functional(self, X, y, model_file=None, fit_params=None):
        pass

    def predict(self, X):
        if self.binary:
            return self.model.predict(X, verbose=True, batch_size=32)
        else:
            return self.model.predict_classes(X, verbose=True, batch_size=32)

    def test_sequential(self, X, y):

        print("Getting predictions on the test set")
        predictions = self.predict(X)
        if self.binary:

            # TODO: Obtain other metrics
            acc = ((predictions.ravel() > 0.5) == (y > 0.5)).mean()

            print("Test set accuracy of {}%.".format(acc * 100.0))
            print("Test set error of {}%.".format((1 - acc) * 100.0))

        else:
            # TODO: Obtain other metrics
            acc = (predictions == y).mean()

        return acc

    def log_results(self, logfile):

        def print_history(history_obj, f):
            for (i, (loss, val_loss)) in enumerate(zip(history_obj.history['loss'],
                                                   history_obj.history['val_loss'])):
                print("Epoch %d: loss: %f, val_loss: %f" % (i+1, loss, val_loss), file=f)

        with open(logfile, "w") as f:
            print("Started training: " + self.ttime_start, file=f)
            print("Stopped training: " + self.ttime_stop, file=f)

            # TODO: include metrics

            print("==" * 40, file=f)
            print("Model: ", file=f)
            print(self.model.to_json(), file=f)
            print("==" * 40, file=f)
            print("Training history:", file=f)
            print_history(self.model.model.history, f)

class RNNClassifier(LanguageClassifier):

    def __init__(self, lembedding, num_classes=2, unit='gru', rnn_size=128, train_vectors=True):

        if not isinstance(lembedding, OneLevelEmbedding):
            raise LanguageClassifierException("The model only accepts one-level language embeddings")
        if num_classes < 2:
            raise LanguageClassifierException("Classes must be 2 or more")

        model = self._generate_model(lembedding, num_classes, unit, rnn_size, train_vectors)
        super(RNNClassifier, self).__init__(model, LanguageClassifier.SEQUENTIAL)

    @staticmethod
    def _generate_model(lembedding, num_classes=2, unit='gru', rnn_size=128, train_vectors=True):

        model = Sequential()
        if lembedding.vector_box.W is None:
            emb = Embedding(lembedding.vector_box.size,
                            lembedding.vector_box.vector_dim,
                            W_constraint=None)
        else:
            emb = Embedding(lembedding.vector_box.size,
                            lembedding.vector_box.vector_dim,
                            weights=[lembedding.vector_box.W], W_constraint=None)
        emb.trainable = train_vectors
        model.add(emb)
        if unit == 'gru':
            model.add(GRU(rnn_size))
        else:
            model.add(LSTM(rnn_size))
        model.add(Dropout(0.2))
        if num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
        else:
            model.add(Dense(num_classes, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])

        return model

class RCNNBinaryClassifier(LanguageClassifier):

    def __init__(self, lembedding, etc):
        if not isinstance(lembedding, TwoLevelsEmbedding):
            raise LanguageClassifierException("The model only accepts two-levels language embeddings")