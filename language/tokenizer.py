from spacy.en import English

class Tokenizer(object):

    def tokenize(self, txt):
        raise NotImplementedError()

    def tokenize_by_sentences(self, txt):
        raise NotImplementedError()

class EnglishTokenizer(Tokenizer):

    def __init__(self):
        self.nlp = English()

    def tokenize(self, text):
        """
        Gets tokens from a text in English
        """
        if not isinstance(text, unicode):
            text = unicode(text)

        tokens = [token.lower_ for token in self.nlp(text)]
        return tokens

    def tokenize_by_sentences(self, text):
        """
        Takes a text and returns a list of lists of tokens, where each sublist is a sentence
        """
        if not isinstance(text, unicode):
            text = unicode(text)

        sentences = self.nlp(text).sents
        return [[t.text for t in s] for s in sentences]
