import sys
import cPickle as pickle

from tokenizer import EnglishTokenizer

class LanguageEmbeddingException(Exception):
    pass

class LanguageEmbedding(object):
    """
    General class for a language embedding. It contains a vector box which will
    give the first level vectors of the embedding, which could be words vectors
    or character vectors.
    """

    def __init__(self, vector_box, verbose=True):
        self.vector_box = vector_box
        self.verbose = verbose
        self.data = None
        self.labels = None

    def save(self, file):
        """
        Saves the object, containing the precomputed text indexed data and
        initial vectors in the vector-box
        """
        if self.data is None:
            raise LanguageEmbeddingException("No text was treated as embedding input, call compute()")
        pickle.dump(self, open(file, 'wb'), pickle.HIGHEST_PROTOCOL)

    def set_labels(self, y):
        """
        Convenience method to also save the labels corresponding to the data saved
        in the embedding object
        """
        self.labels = y

class OneLevelEmbedding(LanguageEmbedding):
    """
    Language embedding consisting of a representation in one level. This is,
    text will be represented as a sequence of tokens (which normally will be words
    or characters) together with a vector box encapsulating the vector representation
    of those tokens.

    For instance, in the case of words, we can have this text:

          "Cervantes was born in a beautiful city of Spain"

    which, by using the word vectors representation found in the corresponding box,
    will be translated and saved in the language embedding as

          [234, 10, 23, 2, 5, 294, 392, 6, 1024]

    and hence ready as input for a model accepting word embeddings.
    """

    CHAR_EMBEDDING = 1
    WORD_EMBEDDING = 2

    def __init__(self, vector_box, size, type=WORD_EMBEDDING, verbose=True):
        super(OneLevelEmbedding, self).__init__(vector_box, verbose)
        self.type = type
        self.size = size

    def compute(self, texts):
        if self.type == OneLevelEmbedding.WORD_EMBEDDING:
            self.data = self._to_word_level_idx(texts)
        elif self.type == OneLevelEmbedding.CHAR_EMBEDDING:
            self.data = self._to_char_level_idx(texts)

    @staticmethod
    def load(file):
        obj = pickle.load(open(file, 'rb'))
        if isinstance(obj, OneLevelEmbedding):
            return obj
        else:
            raise LanguageEmbeddingException("Pickle is not of the expected class")

    def _to_word_level_idx(self, texts, tokenizer=None):
        """
        Receives a list of texts. For each text, it converts the text into indices of a word
        vector container (Glove, WordToVec) for later use in the embedding of a neural network.
        Texts are padded (or reduced) up to the number of words per text given by the size attribute

        The method accepts a user-specific tokenizer to break texts into tokens. If not specified,
        a default English tokenizer is used
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
            tokenized_texts.append(tokenizer.tokenize(text))

        texts_with_indices = normalize(self.vector_box.get_indices(tokenized_texts), self.size)
        return texts_with_indices

    def _to_char_level_idx(self, texts):
        """
        TODO
        """
        texts_characters = []
        for (i, text) in enumerate(texts):
            if self.verbose:
                sys.stdout.write('Processing text %d out of %d \r' % (i + 1, len(texts)))
                sys.stdout.flush()
            texts_characters.append(self.vector_box.get_indices(text))

        texts_characters = normalize(texts_characters, self.size)
        return texts_characters

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