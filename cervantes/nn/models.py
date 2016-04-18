from __future__ import print_function
from time import strftime

from keras.layers.recurrent import GRU, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers.core import Dense, Dropout
import keras.utils.np_utils
import keras.models
import numpy as np

from cervantes.language import OneLevelEmbedding, TwoLevelsEmbedding


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

    def train(self, X, y, nb_epoch=20, validation_split=0.15, batch_size=32,
              model_weights_file=None, model_spec_file=None, **kwargs):
        if self.type == LanguageClassifier.SEQUENTIAL:
            self.train_sequential(X, y, nb_epoch, validation_split, batch_size,
                                  model_weights_file, model_spec_file, **kwargs)
        else:
            self.train_functional(X, y, nb_epoch, validation_split, batch_size,
                                  model_weights_file, model_spec_file, **kwargs)

    def train_sequential(self, X, y, nb_epoch=20, validation_split=0.15,
                         batch_size=32, model_weights_file=None, model_spec_file=None, **kwargs):

        # Provide some default values as training options
        fit_params = {
            "batch_size": batch_size,
            "nb_epoch": nb_epoch,
            "verbose": 1,
            "validation_split": validation_split,
            "callbacks": [EarlyStopping(verbose=True, patience=5, monitor='val_acc')]
        }

        # Add a callback for saving temporal status of the model while training
        if model_weights_file is not None:
            fit_params["callbacks"].append(ModelCheckpoint(model_weights_file, monitor='val_acc',
                                                           verbose=True, save_best_only=True))

        # Override any params provided by the user for Keras training. This allows overriding
        # default Cervantes behavior when training
        fit_params.update(kwargs)

        if max(y) > 1:
            Y = keras.utils.np_utils.to_categorical(y)
            self.binary = False
        else:
            Y = y
            self.binary = True

        try:
            print("Fitting! Hit CTRL-C to stop early...")
            self.ttime_start = strftime("%Y-%m-%d %H:%M:%S")
            self.model.fit(X, Y, **fit_params)
            print("Training finished")
        except KeyboardInterrupt:
            print("Training stopped early!")
        self.ttime_stop = strftime("%Y-%m-%d %H:%M:%S")

        # If weights file was specified, we load the best model computed up to the moment
        # (otherwise the last epoch model would still be working)
        if model_weights_file is not None:
            print("Loading best model found...")
            self.model.load_weights(model_weights_file)

        # Save the full model (spec + weights) if all the information was provided
        if model_weights_file is not None and model_spec_file is not None:
            print("Saving full model...")
            self.save_model(model_spec_file, model_weights_file)

    def train_functional(self, X, y, model_file=None, fit_params=None):
        pass

    def predict(self, X):
        predictions = self.model.predict_classes(X, verbose=True, batch_size=32)
        if len(predictions.shape) > 1:
            return np.array([x[0] for x in predictions])
        return predictions

    def test_sequential(self, X, y, verbose=True):

        if verbose:
            print("Getting predictions on the test set")
        predictions = self.predict(X)
        if len(predictions) != len(y):
            raise LanguageClassifierException("Non comparable arrays")
        if self.binary:
            acc = ((predictions == y)*1.0).mean()
            prec = np.sum(np.bitwise_and(predictions, y))*1.0/np.sum(predictions)
            recall = np.sum(np.bitwise_and(predictions, y))*1.0/np.sum(y)
            if verbose:
                print("Test set accuracy of {0:.3f}%".format(acc * 100.0))
                print("Test set error of {0:.3f}%".format((1 - acc) * 100.0))
                print("Precision for class=1: {0:.3f}".format(prec))
                print("Recall for class=1: {0:.3f}".format(recall))

            return (acc, prec, recall)
        else:
            # TODO: Obtain more metrics for the multiclass problem
            acc = ((predictions == y)*1.0).mean()
            if verbose:
                print("Test set accuracy of {0:.3f}%".format(acc * 100.0))
                print("Test set error of {0:.3f}%".format((1 - acc) * 100.0))
            return acc

    def log_results(self, logfile, X_test, y_test):

        def print_history(history_obj, f):
            for (i, (loss, val_loss, acc, val_acc)) in enumerate(zip(history_obj.history['loss'],
                                                       history_obj.history['val_loss'],
                                                       history_obj.history['acc'],
                                                       history_obj.history['val_acc'])):
                print("Epoch %d: loss: %f, val_loss: %f, acc: %f, val_acc: %f" %
                      (i+1, loss, val_loss, acc, val_acc), file=f)

        with open(logfile, "w") as f:
            print("Started training: " + self.ttime_start, file=f)
            print("Stopped training: " + self.ttime_stop, file=f)

            print("Obtaining test error...")
            results = self.test_sequential(X_test, y_test, verbose=False)
            if self.binary:
                (acc, prec, recall) = results
                print("Test set accuracy of {0:.3f}%".format(acc * 100.0), file=f)
                print("Test set error of {0:.3f}%".format((1 - acc) * 100.0), file=f)
                print("Precision for class=1: {0:.3f}".format(prec), file=f)
                print("Recall for class=1: {0:.3f}".format(recall), file=f)
            else:
                acc = results
                print("Test set accuracy of {0:.3f}%".format(acc * 100.0), file=f)
                print("Test set error of {0:.3f}%".format((1 - acc) * 100.0), file=f)

            print("==" * 40, file=f)
            print("Model: ", file=f)
            print(self.model.to_json(), file=f)
            print("==" * 40, file=f)
            print("Training history:", file=f)
            if self.model.model.history is not None:
                print_history(self.model.model.history, f)
            else:
                print("Training history not available in loaded models", file=f)

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
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

        return model

class RRNNBasicClassifier(LanguageClassifier):

    def __init__(self, lembedding, num_classes=2, rnn='gru', train_vectors=True):
        if not isinstance(lembedding, TwoLevelsEmbedding):
            raise LanguageClassifierException("The model only accepts two-levels language embeddings")

        # TODO

    @staticmethod
    def _generate_model(lembedding, num_classes=2, unit='gru', rnn_size=128, train_vectors=True):


        input_shape = lembedding.size_level1 * lembedding.size_level2
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
