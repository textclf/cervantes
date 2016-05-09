from __future__ import print_function
from time import strftime

from keras.layers.recurrent import GRU, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Input, \
                         Embedding, \
                         merge, \
                         Lambda, \
                         Reshape, \
                         Highway, \
                         TimeDistributed

from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution1D, MaxPooling1D
import keras.backend as K
import keras.utils.np_utils
import keras.models
import numpy as np

from cervantes.language import OneLevelEmbedding, TwoLevelsEmbedding


class LanguageClassifierException(Exception):
    pass

class LanguageClassifier(object):

    def __init__(self, model, optimizer=None):
        self.model = model
        self.binary = "unk"  # unknown until training

        self.ttime_start = None
        self.ttime_stop = None      # for training logging purposes
        self.optimizer = optimizer  # used for saving / loading models

    def save_model(self, model_spec_file, model_weights_file):
        from json import dump, loads

        dump({
                'model': loads(self.model.to_json()),
                'binary': self.binary,
                'optimizer': self.optimizer
            }, 
            open(model_spec_file, 'w'))

        self.model.save_weights(model_weights_file, overwrite=True)

    @staticmethod
    def load_model(model_spec_file, model_weights_file):
        from json import dumps, load

        params = load(open(model_spec_file, "r"))

        model = keras.models.model_from_json(dumps(params['model']))
        binary = params['binary']
        optimizer = params['optimizer']

        model.load_weights(model_weights_file)
        if binary:
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
        else:
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
        
        lc = LanguageClassifier(model)
        lc.binary = binary
        return lc

    def train(self, X, y, nb_epoch=20, validation_split=0.15, batch_size=32,
              model_weights_file=None, model_spec_file=None, **kwargs):

        # Provide some default values as training options
        fit_params = {
            "batch_size": batch_size,
            "nb_epoch": nb_epoch,
            "verbose": 1,
            "validation_split": validation_split,
            "callbacks": [EarlyStopping(verbose=True, patience=5, monitor='val_acc')]
        }

        # Override any params provided by the user for Keras training. This allows overriding
        # default Cervantes behavior when training
        fit_params.update(kwargs)

        # Add a callback for saving temporal status of the model while training
        if model_weights_file is not None:
            fit_params["callbacks"].append(ModelCheckpoint(model_weights_file, monitor='val_acc',
                                                           verbose=True, save_best_only=True))

        # in Keras 1.0, a shape of tuple => single in/out model
        if type(self.model.output_shape) is tuple:
            if max(y) > 1:
                Y = keras.utils.np_utils.to_categorical(y)
                self.binary = False
            else:
                Y = np.array(y)
                self.binary = True
        else:
            raise LanguageClassifierException('Mult-output models are not supported yet')

        # if we don't need 3d inputs...
        if type(self.model.input_shape) is tuple:
            X = np.array(X)
            if len(self.model.input_shape) == 2:
                X = X.reshape((X.shape[0], -1))
        else:
            raise LanguageClassifierException('Mult-input models are not supported yet')

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

    def predict(self, X):
        if type(self.model.input_shape) is tuple:
            X = np.array(X)
            if len(self.model.input_shape) == 2:
                X = X.reshape((X.shape[0], -1))
        else:
            raise LanguageClassifierException('Mult-input models are not supported yet')

        predictions = self.model.predict(X, verbose=True, batch_size=32)
        if (len(predictions.shape) > 1) and (1 not in predictions.shape):
            predictions = predictions.argmax(axis=-1)
        else:
            predictions = 1 * (predictions > 0.5).ravel()
        return predictions

    def predict_proba(self, X):
        if type(self.model.input_shape) is tuple:
            X = np.array(X)
            if len(self.model.input_shape) == 2:
                X = X.reshape((X.shape[0], -1))
        else:
            raise LanguageClassifierException('Mult-input models are not supported yet')
        return self.model.predict(X, verbose=True, batch_size=32)

    def test(self, X, y, verbose=True):
        # if we don't need 3d inputs...
        if type(self.model.input_shape) is tuple:
            X = np.array(X)
            if len(self.model.input_shape) == 2:
                X = X.reshape((X.shape[0], -1))
        else:
            raise LanguageClassifierException('Mult-input models are not supported yet')

        if verbose:
            print("Getting predictions on the test set")
        predictions = self.predict(X)

        if len(predictions) != len(y):
            raise LanguageClassifierException("Non comparable arrays")

        if self.binary:
            acc = (predictions == y).mean()
            prec = np.sum(np.bitwise_and(predictions, y)) * 1.0 / np.sum(predictions)
            recall = np.sum(np.bitwise_and(predictions, y)) * 1.0 / np.sum(y)
            if verbose:
                print("Test set accuracy of {0:.3f}%".format(acc * 100.0))
                print("Test set error of {0:.3f}%".format((1 - acc) * 100.0))
                print("Precision for class=1: {0:.3f}".format(prec))
                print("Recall for class=1: {0:.3f}".format(recall))

            return (acc, prec, recall)
        else:
            # TODO: Obtain more metrics for the multiclass problem
            acc = (predictions == y).mean()
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
            results = self.test(X_test, y_test, verbose=False)
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
            hist = self.model.history \
                   if hasattr(self.model, 'history') \
                   else \
                   self.model.model.history

            if hist is not None:
                print_history(hist, f)
            else:
                print("Training history not available in loaded models", file=f)

class RNNClassifier(LanguageClassifier):

    def __init__(self, lembedding, num_classes=2, unit='gru', rnn_size=128, train_vectors=True,
                 optimizer=None):

        if not isinstance(lembedding, OneLevelEmbedding):
            raise LanguageClassifierException("The model only accepts one-level language embeddings")
        if num_classes < 2:
            raise LanguageClassifierException("Classes must be 2 or more")

        self.optimizer = optimizer
        model = self._generate_model(lembedding, num_classes, unit,
                                     rnn_size, train_vectors)
        super(RNNClassifier, self).__init__(model, self.optimizer)

    def _generate_model(self, lembedding, num_classes=2, unit='gru', rnn_size=128, train_vectors=True):

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
            if self.optimizer is None:
                self.optimizer = 'rmsprop'
            model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=["accuracy"])
        else:
            if self.optimizer is None:
                self.optimizer = 'adam'
            model.add(Dense(num_classes, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=["accuracy"])

        return model

class BiRNNClassifier(LanguageClassifier):

    def __init__(self, lembedding, num_classes=2, unit='gru', rnn_size=128, train_vectors=True,
                 optimizer=None):

        if not isinstance(lembedding, OneLevelEmbedding):
            raise LanguageClassifierException("The model only accepts one-level language embeddings")
        if num_classes < 2:
            raise LanguageClassifierException("Classes must be 2 or more")

        self.optimizer = optimizer
        model = self._generate_model(lembedding, num_classes, unit,
                                     rnn_size, train_vectors)
        super(BiRNNClassifier, self).__init__(model, self.optimizer)

    def _generate_model(self, lembedding, num_classes=2, unit='gru', rnn_size=128, train_vectors=True):

        input = Input(shape=(lembedding.size,), dtype='int32')
        if lembedding.vector_box.W is None:
            emb = Embedding(lembedding.vector_box.size,
                            lembedding.vector_box.vector_dim,
                            W_constraint=None)(input)
        else:
            emb = Embedding(lembedding.vector_box.size,
                            lembedding.vector_box.vector_dim,
                            weights=[lembedding.vector_box.W], W_constraint=None, )(input)
        emb.trainable = train_vectors
        if unit == 'gru':
            forward = GRU(rnn_size)(emb)
            backward = GRU(rnn_size, go_backwards=True)(emb)
        else:
            forward = LSTM(rnn_size)(emb)
            backward = LSTM(rnn_size, go_backwards=True)(emb)

        merged_rnn = merge([forward, backward], mode='concat')
        dropped = Dropout(0.5)(merged_rnn)
        if num_classes == 2:
            out = Dense(1, activation='sigmoid')(dropped)
            model = Model(input=input, output=out)
            if self.optimizer is None:
                self.optimizer = 'rmsprop'
            model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=["accuracy"])
        else:
            out = Dense(num_classes, activation='softmax')(dropped)
            model = Model(input=input, output=out)
            if self.optimizer is None:
                self.optimizer = 'adam'
            model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=["accuracy"])

        return model

class BasicCNNClassifier(LanguageClassifier):

    def __init__(self, lembedding, num_classes=2, num_features=128, train_vectors=True,
                 optimizer=None):

        if not isinstance(lembedding, OneLevelEmbedding):
            raise LanguageClassifierException("The model only accepts one-level language embeddings")
        if num_classes < 2:
            raise LanguageClassifierException("Classes must be 2 or more")

        self.optimizer = optimizer
        model = self._generate_model(lembedding, num_classes, num_features, train_vectors)
        super(BasicCNNClassifier, self).__init__(model, self.optimizer)

    def _generate_model(self, lembedding, num_classes=2, num_features=128, train_vectors=True):

        model = Sequential()
        if lembedding.vector_box.W is None:
            emb = Embedding(lembedding.vector_box.size,
                            lembedding.vector_box.vector_dim,
                            W_constraint=None,
                            input_length=lembedding.size)
        else:
            emb = Embedding(lembedding.vector_box.size,
                            lembedding.vector_box.vector_dim,
                            weights=[lembedding.vector_box.W], W_constraint=None,
                            input_length=lembedding.size)
        emb.trainable = train_vectors
        model.add(emb)

        model.add(Convolution1D(num_features, 3, init='uniform'))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.25))

        model.add(Convolution1D(num_features, 3, init='uniform'))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.25))

        model.add(Flatten())

        if num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            if self.optimizer is None:
                self.optimizer = 'rmsprop'
            model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=["accuracy"])
        else:
            if self.optimizer is None:
                self.optimizer = 'adam'
            model.add(Dense(num_classes, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=["accuracy"])

        return model

class KimCNNClassifier(LanguageClassifier):

    def __init__(self, lembedding, num_classes=2, ngrams=[1,2,3,4,5], nfilters=64, train_vectors=True,
                 optimizer=None):

        if not isinstance(lembedding, OneLevelEmbedding):
            raise LanguageClassifierException("The model only accepts one-level language embeddings")
        if num_classes < 2:
            raise LanguageClassifierException("Classes must be 2 or more")

        self.optimizer = optimizer
        model = self._generate_model(lembedding, num_classes, ngrams,
                                     nfilters, train_vectors)
        super(KimCNNClassifier, self).__init__(model, self.optimizer)

    def _generate_model(self, lembedding, num_classes=2, ngrams=[1,2,3,4,5],
                        nfilters=64, train_vectors=True):

        def sub_ngram(n):
            return Sequential([
                Convolution1D(nfilters, n,
                      activation='relu',
                      input_shape=(lembedding.size, lembedding.vector_box.vector_dim)),
                Lambda(
                    lambda x: K.max(x, axis=1),
                    output_shape=(nfilters,)
                )
        ])

        doc = Input(shape=(lembedding.size, ), dtype='int32')
        embedded = Embedding(input_dim=lembedding.vector_box.size,
                             output_dim=lembedding.vector_box.vector_dim,
                             weights=[lembedding.vector_box.W])(doc)
        embedded.trainable = train_vectors

        rep = Dropout(0.5)(
            merge(
                [sub_ngram(n)(embedded) for n in ngrams],
                mode='concat',
                concat_axis=-1
            )
        )

        if num_classes == 2:
            out = Dense(1, activation='sigmoid')(rep)
            model = Model(input=doc, output=out)
            if self.optimizer is None:
                self.optimizer = 'rmsprop'
            model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=["accuracy"])
        else:
            out = Dense(num_classes, activation='softmax')(rep)
            model = Model(input=doc, output=out)
            if self.optimizer is None:
                self.optimizer = 'adam'
            model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=["accuracy"])

        return model

class DeepCNNClassifier(LanguageClassifier):

    def __init__(self, lembedding, num_classes=2, first_kernel_size=3, num_features=1024, train_vectors=True,
                 optimizer=None, conv_dropout=False):
        ## Original kernel size: 3 for words, 7 for chars as described in LeCunn

        if not isinstance(lembedding, OneLevelEmbedding):
            raise LanguageClassifierException("The model only accepts one-level language embeddings")
        if num_classes < 2:
            raise LanguageClassifierException("Classes must be 2 or more")

        self.optimizer = optimizer
        model = self._generate_model(lembedding, num_classes, first_kernel_size,
                                     num_features, conv_dropout, train_vectors)
        super(DeepCNNClassifier, self).__init__(model, self.optimizer)

    def _generate_model(self, lembedding, num_classes=2, first_kernel_size=3,
                        num_features=1024, conv_dropout=False, train_vectors=True):

        model = Sequential()
        if lembedding.vector_box.W is None:
            emb = Embedding(lembedding.vector_box.size,
                            lembedding.vector_box.vector_dim,
                            W_constraint=None,
                            input_length=lembedding.size)
        else:
            emb = Embedding(lembedding.vector_box.size,
                            lembedding.vector_box.vector_dim,
                            weights=[lembedding.vector_box.W], W_constraint=None,
                            input_length=lembedding.size)
        emb.trainable = train_vectors
        model.add(emb)

        # Two conv layers with original kernel size, maxpooling is 2
        model.add(Convolution1D(num_features, first_kernel_size, init='uniform'))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(2))
        if conv_dropout:
            model.add(Dropout(0.25))

        model.add(Convolution1D(num_features, first_kernel_size, init='uniform'))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(2))
        if conv_dropout:
            model.add(Dropout(0.25))

        # Three conv layers with kernel size = 3, no maxpooling
        model.add(Convolution1D(num_features, 3, init='uniform'))
        model.add(Activation('relu'))
        if conv_dropout:
            model.add(Dropout(0.25))

        model.add(Convolution1D(num_features, 3, init='uniform'))
        model.add(Activation('relu'))
        if conv_dropout:
            model.add(Dropout(0.25))

        model.add(Convolution1D(num_features, 3, init='uniform'))
        model.add(Activation('relu'))
        if conv_dropout:
            model.add(Dropout(0.25))

        # One final conv layer with maxpooling
        model.add(Convolution1D(num_features, 3, init='uniform'))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.25))

        model.add(Flatten())

        # Two dense layers with heavy dropout
        model.add(Dense(2048))
        model.add(Dropout(0.5))

        model.add(Dense(2048))
        model.add(Dropout(0.5))

        if num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            if self.optimizer is None:
                self.optimizer = 'rmsprop'
            model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=["accuracy"])
        else:
            if self.optimizer is None:
                self.optimizer = 'rmsprop'
            model.add(Dense(num_classes, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=["accuracy"])

        return model


class RCNNClassifier(LanguageClassifier):

    def __init__(self, lembedding, num_classes=2, ngrams=[1, 2, 3, 4, 5],
                 nfilters=64, rnn_type=GRU, rnn_dim=80, train_vectors=True,
                 optimizer=None):

        if not isinstance(lembedding, TwoLevelsEmbedding):
            raise LanguageClassifierException(
                "The model only accepts two-level language embeddings")
        if num_classes < 2:
            raise LanguageClassifierException("Classes must be 2 or more")

        self.optimizer = optimizer
        model = self._generate_model(lembedding, num_classes, ngrams,
                                     nfilters, rnn_type, rnn_dim, train_vectors)
        super(RCNNClassifier, self).__init__(model, self.optimizer)

    def _generate_model(self, lembedding, num_classes=2, ngrams=[1,2,3,4,5],
                        nfilters=64, rnn_type=GRU, rnn_dim=80, train_vectors=True):

        CHARACTERS_PER_WORD = lembedding.size_level1
        WORDS_PER_DOCUMENT = lembedding.size_level2
        EMBEDDING_DIM = lembedding.vector_box.vector_dim

        INPUT_SHAPE = (CHARACTERS_PER_WORD * WORDS_PER_DOCUMENT, )
        EMBEDDING_SHAPE = (WORDS_PER_DOCUMENT, CHARACTERS_PER_WORD, EMBEDDING_DIM)

        doc = Input(shape=(INPUT_SHAPE[0], ), dtype='int32')

        embedded = Sequential([
                Embedding(
                    input_dim=lembedding.vector_box.size, 
                    output_dim=EMBEDDING_DIM, 
                    input_length=INPUT_SHAPE[0]
                    ), 
                Reshape(EMBEDDING_SHAPE)
            ])(doc)

        def sub_model(n):
            return Sequential([
                    Convolution1D(nfilters, n, 
                        activation='relu', 
                        input_shape=EMBEDDING_SHAPE[1:]
                        ), 
                    Lambda(
                        lambda x: K.max(x, axis=1), 
                        output_shape=(nfilters,)
                        )
                ])

        rep = Dropout(0.5)(
            merge(
                [TimeDistributed(sub_model(n))(embedded) for n in ngrams], 
                mode='concat', 
                concat_axis=-1
            )
        )

        out = Dropout(0.5)(
            merge(
                [rnn_type(rnn_dim)(rep), rnn_type(rnn_dim, go_backwards=True)(rep)], 
                mode='concat', 
                concat_axis=-1
            )
        )

        mapping = [
                Highway(activation='relu'),
                Dropout(0.5),
                Dense(64, activation='relu'),
                Dropout(0.4)
            ]

        for f in mapping:
            out = f(out)

        if num_classes == 2:
            out = Dense(1, activation='sigmoid')(out)
            model = Model(input=doc, output=out)
            if self.optimizer is None:
                self.optimizer = 'rmsprop'
            model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=["accuracy"])
        else:
            out = Dense(num_classes, activation='softmax')(out)
            model = Model(input=doc, output=out)
            if self.optimizer is None:
                self.optimizer = 'adam'
            model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=["accuracy"])
        
        return model


class RRNNClassifier(LanguageClassifier):

    def __init__(self, lembedding, num_classes=2, rnn_dim=32, optimizer=None):

        if not isinstance(lembedding, TwoLevelsEmbedding):
            raise LanguageClassifierException(
                "The model only accepts two-level language embeddings")
        if num_classes < 2:
            raise LanguageClassifierException("Classes must be 2 or more")

        self.optimizer = optimizer
        model = self._generate_model(lembedding, num_classes, rnn_dim)
        super(RRNNClassifier, self).__init__(model, self.optimizer)

    def _generate_model(self, lembedding, num_classes=2, rnn_dim=32):

        WORD_PER_SENTENCES = lembedding.size_level1
        SENTENCES_PER_DOCUMENT = lembedding.size_level2
        EMBEDDING_DIM = lembedding.vector_box.vector_dim

        INPUT_SHAPE = (WORD_PER_SENTENCES * SENTENCES_PER_DOCUMENT, )
        EMBEDDING_SHAPE = (SENTENCES_PER_DOCUMENT, WORD_PER_SENTENCES, EMBEDDING_DIM)

        doc = Input(shape=(INPUT_SHAPE[0], ), dtype='int32')

        embedded = Sequential([
            Embedding(
                input_dim=lembedding.vector_box.size,
                output_dim=EMBEDDING_DIM,
                input_length=INPUT_SHAPE[0]
            ),
            Reshape(EMBEDDING_SHAPE)
        ])(doc)

        out = TimeDistributed(GRU(rnn_dim))(embedded)
        next = Dropout(0.5)(out)
        out = GRU(rnn_dim)(next)
        out = Dropout(0.5)(out)

        mapping = [
            Dense(64, activation='relu'),  # Maybe add more layers
        ]

        for f in mapping:
            out = f(out)

        if num_classes == 2:
            out = Dense(1, activation='sigmoid')(out)
            model = Model(input=doc, output=out)
            if self.optimizer is None:
                self.optimizer = 'rmsprop'
            model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=["accuracy"])
        else:
            out = Dense(num_classes, activation='softmax')(out)
            model = Model(input=doc, output=out)
            if self.optimizer is None:
                self.optimizer = 'adam'
            model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=["accuracy"])

        return model
