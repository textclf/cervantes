from spacy.en import English
import re

class Tokenizer(object):

    def tokenize(self, txt):
        """
        Returns a list of tokens from a list
        """
        raise NotImplementedError()

    def tokenize_by_sentences(self, txt):
        """
        Takes a text and returns a list of lists of tokens, where each sublist is a sentence
        """
        raise NotImplementedError()

class EnglishTokenizer(Tokenizer):
    """
    English fully-fledged tokenization performed by spaCy.
    """

    def __init__(self):
        self.nlp = English()

    def tokenize(self, text):
        if not isinstance(text, unicode):
            text = unicode(text, errors='ignore')

        tokens = [token.lower_ for token in self.nlp(text)]
        return tokens

    def tokenize_by_sentences(self, text):
        if not isinstance(text, unicode):
            text = unicode(text, errors='ignore')

        sentences = self.nlp(text).sents
        return [[t.text for t in s] for s in sentences]
    ## TODO: Watch out for lower case

class FastTokenizer(Tokenizer):
    """
    Fast and simple tokenizer based on a given regular expression. By default, it
    accepts tokens of two or more alphanumeric characters and ignores punctuation.
    This is the same default tokenizer present in sklearn.

    Note that while this is a fine tokenizer for bag of words models, the nuances lost
    by not considering punctuation and more fine-grained tokens might have an
    impact in the more elaborate deep learning models. Except for proofs of concept,
    the fully-fledged tokenization performed by spaCy is recommended and the default
    in Cervantes.
    """

    def __init__(self, token_pattern=r"(?u)\b\w\w+\b"):
        self.tokenizer = re.compile(token_pattern)

    def tokenize(self, text):
        if not isinstance(text, unicode):
            text = unicode(text, errors='ignore')

        words = self.tokenizer.findall(text)
        return [word.lower() for word in words]
