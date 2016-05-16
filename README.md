# Cervantes: Deep Learning library for text classification

## Status: library working but still under heavy development / refactorization

Cervantes is a library built over [Keras](http://keras.io) that makes the process of using deep learning for text classification and sentiment analysis very easy. It includes basic deep architectures using word vectors, as well as complex models combining CNNs and RNNs over text hierarchies. Furthermore, it is built as a general framework where more advanced users can quickly experiment with new models without having to worry about data preprocessing.

Some of the features offered by Cervantes:

- allows creating models based on character and word vectors. 
- provides easily configurable data preprocessing and tokenization (using [spaCy](http://spacy.io)) so that it is easy to convert a text into an embedding that a neural network can use as input.
- caters for any kind of users. Cervantes offers performant default classifiers that can be trained without any knowledge of Keras or deep learning. However, advanced users can easily create their own Keras models and connect them to the framework.
-  allows for complex two-level hierarchical embeddings, computing word-sentence-text or character-word-text representations.

## Quick example

Here is a quick example showing how to train a bidirectional recurrent neural network for text classification:

```python
from cervantes.box import WordVectorBox
from cervantes.language import OneLevelEmbedding
from cervantes.nn.models import BiRNNClassifier

# Create and build a word vector container from pre-computed vectors
vbox = WordVectorBox("word_vectors.txt")
vbox.build()

# Transform the training texts into a language embedding consisting of word vectors
lang_embedding = OneLevelEmbedding(vbox, size=200)
lang_embedding.compute_word_repr(train_texts + test_texts)
lang_embedding.set_labels(train_labels + test_labels)

# Instantiate and train a model using a bidirectional GRU over sequences of word vectors
clf = BiRNNClassifier(lang_embedding, num_classes=2, unit='gru', 
                      rnn_size=64, train_vectors=True)
clf.train(X=lembedding.data[:len(train_labels)],
          y=lembedding.labels[:len(train_labels)])
```
 

------------------