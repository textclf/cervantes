import string

from cervantes.box.vbox import VectorBox

class CharBox(VectorBox):
    """
    Generic class for mapping characters to indexes for character embeddings.
    The class implicitly defines a relation character to vector in a embedding
    by uniquely identifying each character with an index number.

    Subclasses can treat different languages and sets of characters.
    """

    def __init__(self, vector_dim):
        super(CharBox, self).__init__()
        self.vector_dim = vector_dim
        self._set_size()

    def _set_size(self):
        """ Subclass must specify the number contained in the box"""
        raise NotImplementedError()

    def i2c(self, i):
        """ Index to character. """
        raise NotImplementedError()

    def c2i(self, i):
        """ Character to index. """
        raise NotImplementedError()

    def __getitem__(self, obj):
        """ Easy general access, o can be index or string"""
        raise NotImplementedError()

    def get_indices(self, obj):
        return self.__getitem__(obj)

class EnglishCharBox(CharBox):
    """
    Specific CharBox for standard English characters and punctuation with no upper-case
    and lower-case distinction

    The argument mark_limits specifies whether or not special tokens should be added
    to indicate beginning and end of block. Main use of this is to explicitly indicate
    beginning and end of words.

    Example:
    ---------
    >>> cm = EnglishCharBox(25, mark_limits=True)
    >>> cm[['My dog.', 'My cat.']]
    [[70, 23, 35, 69, 14, 25, 17, 50, 71], [70, 23, 35, 69, 13, 11, 30, 50, 71]]
    >>> cm[cm[['My dog.', 'My cat.']]]
    [['<block>', 'm', 'y', ' ', 'd', 'o', 'g', '.', '</block>'],
     ['<block>', 'm', 'y', ' ', 'c', 'a', 't', '.', '</block>']]
    """

    ALLOWED_CHARS = [ch for ch in (string.digits + string.lowercase + string.punctuation + ' ')] +\
                    ['<block>', '</block>']

    def __init__(self, vector_dim, mark_limits=True):
        self.n_chars = len(self.ALLOWED_CHARS)
        self.mark_limits = mark_limits

        self._c2i = {ch: (i + 1) for i, ch in enumerate(self.ALLOWED_CHARS)}
        self._c2i['<unk>'] = self.n_chars + 1
        
        self._i2c = {(i + 1): ch for i, ch in enumerate(self.ALLOWED_CHARS)}
        self._i2c[self.n_chars + 1] = '<unk>'
        self._i2c[0] = '<blank>'

        super(EnglishCharBox, self).__init__(vector_dim)

    def _set_size(self):
        self.size = len(self._i2c)

    def i2c(self, i):
        try:
            return self._i2c[i]
        except KeyError:
            return '<unk>'

    def c2i(self, c):
        try:
            return self._c2i[c]
        except KeyError:
            return self.n_chars + 1

    def __getitem__(self, obj):
        if isinstance(obj, int):
            return self.i2c(obj)
        if isinstance(obj, str) or isinstance(obj, unicode):
            if self.mark_limits:
                return [self._c2i['<block>']] + [self.c2i(ch) for ch in obj.lower()] + [self._c2i['</block>']]
            else:
                return [self.c2i(ch) for ch in obj.lower()]
        if hasattr(obj, '__iter__'):
            return [self.__getitem__(so) for so in obj]
            
    def get_indices(self, obj):
        return self.__getitem__(obj)
