import theano
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
from sklearn.metrics.pairwise import pairwise_distances

class Quantizer(theano.Op):
    """
    Quantizer;
    """
    __props__ = ("sharedBins", )

    itypes = [theano.tensor.fmatrix]
    otypes = [theano.tensor.fmatrix]

    def __init__(self, sharedBins=None):
        self.sharedBins = sharedBins
        super(Quantizer, self).__init__()

    def perform(self, node, inputs, output_storage):
        # Input & output storage settings
        x = inputs[0]
        out = output_storage[0]
        sharedBins = self.sharedBins.get_value()

        if sharedBins == None:
            new_theta = x
        else:
            new_theta = self.quantizeWithBins(theta=x, sharedBins=sharedBins)

        out[0] = new_theta

    """
    sharedBins -> Unique bins
    inputs -> (-1, 6) shaped theta inputs, first dimension is batch
    """
    def quantizeWithBins(self, theta, sharedBins):

        shape = theta.shape
        theta_prime = np.zeros(shape=shape, dtype=theano.config.floatX)

        for i in range(shape[1]):
            # Get Variables
            bins = np.expand_dims(sharedBins[i], axis=0)
            theta_i = np.expand_dims(theta[:, i], axis=0)

            # Calculate distance
            dists = pairwise_distances(bins.T, theta_i.T)

            # Find minimum indexses and set new theta from the bins
            mins = np.argmin(dists, axis=0)
            theta_prime_i = bins.flatten()[mins]

            # Set new theta array
            theta_prime[:, i] = theta_prime_i

        # return Theta'
        return theta_prime

    def grad(self, inputs, output_grads):
        return [output_grads[0]]

if __name__ == "__main__":
    print "Not Implemented Yet"
