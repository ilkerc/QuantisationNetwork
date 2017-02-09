from lasagne.layers import Layer
from helpers.DiscOP import DiscOP
from helpers.Quantizer import Quantizer
import theano.tensor as T


class DiscreteLayer(Layer):

    def __init__(self, incoming, sharedBins, **kwargs):
        super(DiscreteLayer, self).__init__(incoming, **kwargs)
        self.sharedBins = sharedBins
        self.op = Quantizer(sharedBins)

    def get_output_for(self, inputs, **kwargs):
        theta = inputs
        return self.op(theta)


if __name__ == "__main__":
    print "Not Implemented Yet"
