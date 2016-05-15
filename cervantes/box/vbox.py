class VectorBoxException(Exception):
    pass

class VectorBox(object):
    """
    Box of vectors for a series of elements.

    The box has a size (number of elements) and a vector dimension.
    The box will actually contain the vectors in a matrix W (number of elements x vector dim)
    or will implicitly be defined by the number of elements and the vector dimension
    (this is useful if you don't need to initial values for vectors, for instance
    for character vector models, which are usually computed via the neural network model).
    """

    def __init__(self):
        self.W = None        # Vector matrix
        self.num_elems = 0
        self.vdim = 0

    @property
    def size(self):
        if self.W is not None:
            return self.W.shape[0]
        else:
            return self.num_elems

    @size.setter
    def size(self, val):
        if self.W is not None:
            raise VectorBoxException("Size is currently controlled by the shape of W")
        else:
            self.num_elems = val

    @property
    def vector_dim(self):
        if self.W is not None:
            return self.W.shape[1]
        else:
            return self.vdim

    @vector_dim.setter
    def vector_dim(self, val):
        if self.W is not None:
            raise VectorBoxException("Vector dimension is currently controlled by the shape of W")
        else:
            self.vdim = val

