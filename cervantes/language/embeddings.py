import sys
import cPickle as pickle

from tokenizer import EnglishTokenizer
from cervantes.box import VectorBox, WordVectorBox, CharBox

class LanguageEmbeddingException(Exception):
    pass

class LanguageEmbedding(object):
    """
    General class for a language embedding. Language embeddings are objects that
    translate text into vector inputs and have functionality to save and load these
    representations. The values computed can then be used as the input of a neural network.

    A language embedding contains a vector box which holds the translation of text into
    a vector representation given by unique biyections token <-> vector index. Hence,
    when creating a language embedding it is necessary to break down the text into tokens
    (words, characters, etc) and then provide a one to one mapping into vector indexes.

    Parameters
    ----------
    vector_box : vector box object
    verbose: boolean
        if True, information about the computations performed by the LanguageEmbedding
        will be shown
    """

    def __init__(self, vector_box, verbose=True):
        if not isinstance(vector_box, VectorBox):
            raise LanguageEmbeddingException("vector_box must be of class VectorBox")
        self.vector_box = vector_box
        self.verbose = verbose
        self.data = None
        self.labels = None

    def save(self, file):
        """
        Saves the object, containing the precomputed text indexed data and
        initial vectors in the vector-box.
        """
        if self.data is None:
            raise LanguageEmbeddingException("No text was treated as embedding input, call compute()")
        pickle.dump(self, open(file, 'wb'), pickle.HIGHEST_PROTOCOL)

    def set_labels(self, y):
        """
        Convenience method to also save the labels corresponding to the data saved
        in the embedding object.
        """
        self.labels = y

class OneLevelEmbedding(LanguageEmbedding):
    """
    Language embedding consisting of a language representation in one level. This is,
    text will be represented as a sequence of tokens (which normally will be words
    or characters) together with a vector box encapsulating the vector representation
    of those tokens.

    For instance, in the case of words, we can have this text:

          "Cervantes was born in a beautiful city of Spain"

    which, by using the word vectors representation found in the corresponding box,
    will be translated and saved in the language embedding as

          [234, 10, 23, 2, 5, 294, 392, 6, 1024]

    and hence ready as input for a model accepting word embeddings.

    Parameters
    ----------
    vector_box : vector box object
        must hold the translation token <-> vector index. Note that in some cases this
        means that the vector_box has to be built beforehand.
    size: int
        predefined size for each text representation. For the moment, model inputs
        must be of constant size (for instance, 100 words or 200 characters). If the original
        text is smaller than the size, the data is zero-padded. If it is bigger, last
        elements are ignored.
    verbose: boolean
        if True, information about the computations performed by the LanguageEmbedding
        will be shown
    """

    def __init__(self, vector_box, size, verbose=True):
        super(OneLevelEmbedding, self).__init__(vector_box, verbose)
        self.size = size

    @staticmethod
    def load(file):
        obj = pickle.load(open(file, 'rb'))
        if isinstance(obj, OneLevelEmbedding):
            return obj
        else:
            raise LanguageEmbeddingException("Pickle is not of the expected class")

    def compute_index_repr(self, texts, tokenizer=None):
        """
        Converts texts to index representation. The computation process is as follows:
            1st => tokenize the text in tokens as specified in a tokenizer object (if tokenizer
                   is none, the method assumes that no tokenization has to be done).
            2nd => for each token, obtain the corresponding vector index via a vector_box

        Texts are padded (or reduced) up to the number of words per text given by the size attribute

        Parameters
        ----------
        texts: list of texts to compute
        tokenizer: object of class tokenizer that breaks texts into tokens. If none, the text
               is not tokenized (useful for character embeddings, since texts are trivially
               tokenized into characters)
        """
        tokenized_texts = []
        for (i, text) in enumerate(texts):
            if self.verbose:
                sys.stdout.write('Processing text %d out of %d \r' % (i + 1, len(texts)))
                sys.stdout.flush()
            if tokenizer is not None:
                tokenization = tokenizer.tokenize(text)
                tokenized_texts.append(self.vector_box.get_indices(tokenization))
            else:
                tokenized_texts.append(self.vector_box.get_indices(text))
        if self.verbose:
            "Finished processing texts"

        self.data = normalize(tokenized_texts, self.size)

    def compute_word_repr(self, texts, tokenizer=None):
        """
        Convenience method for computing word level embeddings.

        The default tokenizer is the EnglishTokenizer object.
        """
        if not isinstance(self.vector_box, WordVectorBox):
            raise LanguageEmbeddingException("Vector box should be of class WordVectorBox")
        if self.vector_box.W is None:
            raise LanguageEmbeddingException("Word vector box must be built() before")

        if tokenizer is None:
            if self.verbose:
                print "Loading tokenizer"
            tokenizer = EnglishTokenizer()
        self.compute_index_repr(texts, tokenizer)

    def compute_char_repr(self, texts):
        """
        Convenience method for computing char level embeddings
        """
        if not isinstance(self.vector_box, CharBox):
            raise LanguageEmbeddingException("Vector box should be of class CharBox")
        self.compute_index_repr(self, texts)

class TwoLevelsEmbedding(LanguageEmbedding):

    CHAR_WORD_EMBEDDING = 1
    WORD_PARAGRAPH_EMBEDDING = 2

    def __init__(self, vector_box, size_level1, size_level2, type=CHAR_WORD_EMBEDDING, verbose=True):
        super(TwoLevelsEmbedding, self).__init__(vector_box, verbose)
        self.type = type
        self.size_level1 = size_level1
        self.size_level2 = size_level2

    def compute(self, texts):
        if self.type == self.WORD_PARAGRAPH_EMBEDDING:
            self.data = self._to_sentence_level_idx(texts)
        else:
            self.data = self._get_index_repr(self.type, texts)

    @staticmethod
    def load(file):
        obj = pickle.load(open(file, 'rb'))
        if isinstance(obj, TwoLevelsEmbedding):
            return obj
        else:
            raise LanguageEmbeddingException("Pickle is not of the expected class")

    def _to_sentence_level_idx(self, texts, tokenizer=None):
        """
        Receives a list of texts. For each text, it converts the text into sentences and converts the words into
        indices of a word vector container (Glove, WordToVec) for later use in the embedding of a neural network.
        Sentences are padded (or reduced) up to words_per_sentence elements.
        Texts ("paragraphs") are padded (or reduced) up to sentences_per_paragraph
        If prepend = True, padding is added at the beginning
        Ex: [[This might be cumbersome. Hopefully not.], [Another text]]
               to
            [  [[5, 24, 3, 223], [123, 25, 0, 0]]. [[34, 25, 0, 0], [0, 0, 0, 0]  ]
            using sentences_per_paragraph = 4, words_per_sentence = 4
        # words_per_sentence = size_level1
        # sentences_per_text = size_level2
        """
        if tokenizer is None:
            if self.verbose:
                print "Loading tokenizer"
            tokenizer = EnglishTokenizer()

        tokenized_texts = []
        for (i, text) in enumerate(texts):
            if self.verbose:
                sys.stdout.write('Processing text %d out of %d \r' % (i + 1, len(texts)))
                sys.stdout.flush()
            tokenized_texts.append(tokenizer.tokenize_by_sentences(text))

        text_normalized_sentences = [normalize(text, size=self.size_level2)
                                          for text in self.vector_box.get_indices(tokenized_texts)]
        text_normalized_total = normalize(text_normalized_sentences, size=self.size_level1, filler=[0] * self.size_level2)

        return text_normalized_total

    def _get_index_repr(self, level, texts, tokenizer=None):
        if level not in {TwoLevelsEmbedding.CHAR_WORD_EMBEDDING, 
                         TwoLevelsEmbedding.WORD_PARAGRAPH_EMBEDDING}:               
            raise LanguageEmbeddingException(
                'level must be one of CHAR_WORD_EMBEDDING or WORD_PARAGRAPH_EMBEDDING'
            )

        if tokenizer is None:
            if self.verbose:
                print "Loading tokenizer"
            tokenizer = EnglishTokenizer()
        
        _tokenizer = lambda t: tokenizer.tokenize(t) \
                     if level == TwoLevelsEmbedding.CHAR_WORD_EMBEDDING \
                     else \
                     lambda t: tokenizer.tokenize_by_sentences(t)

        tokenized_texts = []
        for (i, text) in enumerate(texts):
            if self.verbose:
                sys.stdout.write('Processing text %d out of %d \r' % (i + 1, len(texts)))
                sys.stdout.flush()
            tokenized_texts.append(_tokenizer(text))

        text_normalized_sentences = [normalize(txt, size=self.size_level1)
                                          for txt in self.vector_box.get_indices(tokenized_texts)]
        text_normalized_total = normalize(
                sq=text_normalized_sentences, 
                size=self.size_level2, 
                filler=[0] * self.size_level1
            )

        return text_normalized_total


def normalize(sq, size=30, filler=0, prepend=False):
    """
    Take a list of lists and ensure that they are all of length `sz`

    Args:
    -----
    e: a non-generator iterable of lists
    sz: integer, the size that each sublist should be normalized to
    filler: obj -- what should be added to fill out the size?
    prepend: should `filler` be added to the front or the back of the list?
    """
    if not prepend:
        def _normalize(e, sz):
            return e[:sz] if len(e) >= sz else e + [filler] * (sz - len(e))
    else:
        def _normalize(e, sz):
            return e[-sz:] if len(e) >= sz else [filler] * (sz - len(e)) + e
    return [_normalize(e, size) for e in sq]