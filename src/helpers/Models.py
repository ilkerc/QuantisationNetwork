import numpy as np
import theano
import lasagne
from DiscreteLayer import DiscreteLayer
from lasagne.init import Constant
from lasagne.layers import ReshapeLayer, DenseLayer, InputLayer, \
    TransformerLayer, Upscale2DLayer, TransposedConv2DLayer, \
    DropoutLayer, TPSTransformerLayer

try:
    from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
    from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer

    print('Using cuda_convnet (faster)')
except ImportError:
    from lasagne.layers import Conv2DLayer as Conv2DLayer
    from lasagne.layers import MaxPool2DLayer as MaxPool2DLayer

    print('Using lasagne.layers (slower)')


def test_model(input_shape):
    b = np.zeros((2, 3), dtype=theano.config.floatX)
    b[0, 0] = 1
    b[1, 1] = 1
    b = b.flatten()
    l_in = InputLayer(shape=(None, input_shape[1], input_shape[2], input_shape[3]))
    l_dense = DenseLayer(l_in, num_units=6, W=lasagne.init.Constant(0.0), b=b)
    l_transform = TransformerLayer(l_in, l_dense, downsample_factor=1)
    return l_transform


def build_mitosis_encoder(input_shape, encoding_size=32, withst=False):
    # Parameters
    filter_size = (3, 3)
    num_filters = 32
    pool_size = (2, 2)
    # Localization Network

    l_input = InputLayer(shape=(None, input_shape[1], input_shape[2], input_shape[3]))
    l_conv1 = Conv2DLayer(l_input, num_filters=num_filters, filter_size=filter_size)
    l_conv2 = Conv2DLayer(l_conv1, num_filters=num_filters, filter_size=filter_size)
    l_pool1 = MaxPool2DLayer(l_conv2, pool_size=pool_size)
    l_pipe1_layer = l_pool1  # We need this

    # ST Network
    if withst:
        # ST Params
        b = np.zeros((2, 3), dtype=theano.config.floatX)
        b[0, 0] = 1
        b[1, 1] = 1
        b = b.flatten()

        # ST Layers
        st_encode1 = DenseLayer(l_pool1, num_units=50, W=lasagne.init.HeUniform('relu'))
        st_encode2 = DenseLayer(st_encode1, num_units=6, b=b, W=lasagne.init.Constant(0.0))
        l_trans1 = TransformerLayer(l_input, st_encode2, downsample_factor=1.0)

        # Localization Network

        st_conv1 = Conv2DLayer(l_trans1, num_filters=num_filters, filter_size=filter_size)
        st_covn2 = Conv2DLayer(st_conv1, num_filters=num_filters, filter_size=filter_size)
        st_pool1 = MaxPool2DLayer(st_covn2, pool_size=pool_size)
        l_pipe1_layer = st_pool1

    # Encoding Step
    l_reshape1 = ReshapeLayer(l_pipe1_layer, shape=([0], -1))
    l_encode = DenseLayer(l_reshape1, num_units=encoding_size, W=lasagne.init.HeUniform('relu'), name='encoder')

    # Decoding Step
    l_decode = DenseLayer(l_encode, W=l_encode.W.T, num_units=l_reshape1.output_shape[1])
    l_reshape2 = ReshapeLayer(l_decode, shape=([0], num_filters,
                                               int(np.sqrt(l_reshape1.output_shape[1] / num_filters)),
                                               int(np.sqrt(l_reshape1.output_shape[1] / num_filters))))

    # Deconv Network
    l_unpool1 = Upscale2DLayer(l_reshape2, scale_factor=pool_size)
    l_deconv2 = TransposedConv2DLayer(l_unpool1,
                                      num_filters=l_conv2.input_shape[1],
                                      W=l_conv2.W,
                                      filter_size=l_conv2.filter_size,
                                      stride=l_conv2.stride,
                                      crop=l_conv2.pad,
                                      flip_filters=not l_conv2.flip_filters)

    l_deconv1 = TransposedConv2DLayer(l_deconv2,
                                      num_filters=l_conv1.input_shape[1],
                                      W=l_conv1.W,
                                      filter_size=l_conv1.filter_size,
                                      stride=l_conv1.stride,
                                      crop=l_conv1.pad,
                                      flip_filters=not l_conv1.flip_filters)

    return l_deconv1


# Spatial Transformer Network with spline
def build_st_spline_network(input_shape):
    W = b = lasagne.init.Constant(0.0)
    num_points = 4
    num_filters = 64
    filter_size = (3, 3)
    pool_size = (2, 2)

    l_in = InputLayer(shape=(None,
                             input_shape[1],
                             input_shape[2],
                             input_shape[3]))

    l_conv1 = Conv2DLayer(l_in,
                          num_filters=num_filters,
                          filter_size=filter_size)

    l_pool1 = MaxPool2DLayer(l_conv1,
                             pool_size=pool_size)

    l_conv2 = Conv2DLayer(l_pool1,
                          num_filters=num_filters,
                          filter_size=filter_size)

    l_pool2 = MaxPool2DLayer(l_conv2,
                             pool_size=pool_size)

    l_dense1 = DenseLayer(l_pool2,
                          num_units=128)

    l_dense2 = DenseLayer(l_dense1,
                          num_units=num_points * 2,
                          W=W,
                          b=b,
                          nonlinearity=None)

    l_st = TPSTransformerLayer(l_in,
                               l_dense2,
                               control_points=num_points)

    l_output = ReshapeLayer(l_st,
                            shape=([0], -1))

    return l_output


# This builds a model of Conv. Autoencoder, Got 2 conv-deconv layers
def build_cnnae_network_2conv(input_shape):
    conv_filters = 16
    filter_size = 3
    pool_size = 2
    encode_size = input_shape[2] * 2

    l_in = InputLayer(shape=(None,
                             input_shape[1],
                             input_shape[2],
                             input_shape[3]))

    l_conv1 = Conv2DLayer(l_in,
                          num_filters=conv_filters,
                          filter_size=(filter_size, filter_size),
                          nonlinearity=None,
                          pad='same')

    l_pool1 = MaxPool2DLayer(l_conv1,
                             pool_size=(pool_size, pool_size))

    l_conv2 = Conv2DLayer(l_pool1,
                          num_filters=conv_filters,
                          filter_size=(filter_size, filter_size),
                          nonlinearity=None,
                          pad='same')

    l_pool2 = MaxPool2DLayer(l_conv2,
                             pool_size=(pool_size, pool_size))

    l_dropout1 = DropoutLayer(l_pool2,
                              p=0.5)

    l_reshape1 = ReshapeLayer(l_dropout1, shape=([0], -1))

    l_encode = DenseLayer(l_reshape1,
                          name='encode',
                          num_units=encode_size)

    l_decode = DenseLayer(l_encode,
                          W=l_encode.W.T,
                          num_units=l_reshape1.output_shape[1])

    l_reshape2 = ReshapeLayer(l_decode,
                              shape=([0],
                                     conv_filters,
                                     int(np.sqrt(l_reshape1.output_shape[1] / conv_filters)),
                                     int(np.sqrt(l_reshape1.output_shape[1] / conv_filters))))

    l_unpool2 = Upscale2DLayer(l_reshape2,
                               scale_factor=pool_size)

    l_deconv2 = TransposedConv2DLayer(l_unpool2,
                                      num_filters=l_conv2.input_shape[1],
                                      W=l_conv2.W,
                                      filter_size=l_conv2.filter_size,
                                      stride=l_conv2.stride,
                                      crop=l_conv2.pad,
                                      flip_filters=not l_conv2.flip_filters)

    l_unpool1 = Upscale2DLayer(l_deconv2,
                               scale_factor=pool_size)

    l_deconv1 = TransposedConv2DLayer(l_unpool1,
                                      num_filters=l_conv1.input_shape[1],
                                      W=l_conv1.W,
                                      filter_size=l_conv1.filter_size,
                                      stride=l_conv1.stride,
                                      crop=l_conv1.pad,
                                      flip_filters=not l_conv1.flip_filters)

    l_output = ReshapeLayer(l_deconv1,
                            shape=([0], -1))

    return l_output


# input_shape = (size, channel, width, height)

def build_st_network(b_size, input_shape, withdisc=True):
    # General Params
    num_filters = 64
    filter_size = (3, 3)
    pool_size = (2, 2)

    # SP Param
    b = np.zeros((2, 3), dtype=theano.config.floatX)
    b[0, 0] = 1
    b[1, 1] = 1
    b = b.flatten()  # identity transform

    # Localization Network
    l_in = InputLayer(shape=(None,
                             input_shape[1],
                             input_shape[2],
                             input_shape[3]))

    l_conv1 = Conv2DLayer(l_in,
                          num_filters=num_filters,
                          filter_size=filter_size)

    l_pool1 = MaxPool2DLayer(l_conv1,
                             pool_size=pool_size)

    l_conv2 = Conv2DLayer(l_pool1,
                          num_filters=num_filters,
                          filter_size=filter_size)

    l_pool2 = MaxPool2DLayer(l_conv2,
                             pool_size=pool_size)

    l_loc = DenseLayer(l_pool2,
                       num_units=64,
                       W=lasagne.init.HeUniform('relu'))

    l_param_reg = DenseLayer(l_loc,
                             num_units=6,
                             b=b,
                             nonlinearity=lasagne.nonlinearities.linear,
                             W=lasagne.init.Constant(0.0),
                             name='param_regressor')

    if withdisc:
        l_dis = DiscreteLayer(l_param_reg, start=Constant(-3.), stop=Constant(3.), linrange=Constant(50.))
    else:
        l_dis = l_param_reg

    # Transformer Network
    l_trans = TransformerLayer(l_in,
                               l_dis,
                               downsample_factor=1.0)

    final = ReshapeLayer(l_trans,
                         shape=([0], -1))
    return final


# Thisbuilds a model of Conv. Autoencoder (Simple 1 layer conv-deconv)
def build_cnnae_network(input_shape):
    conv_filters = 16
    filter_size = 3
    pool_size = 2
    encode_size = input_shape[2] * 2

    l_in = InputLayer(shape=(None,
                             input_shape[1],
                             input_shape[2],
                             input_shape[3]))

    l_conv1 = Conv2DLayer(l_in,
                          num_filters=conv_filters,
                          filter_size=(filter_size, filter_size),
                          nonlinearity=None)

    l_pool1 = MaxPool2DLayer(l_conv1,
                             pool_size=(pool_size, pool_size))

    l_dropout1 = DropoutLayer(l_pool1,
                              p=0.5)

    l_reshape1 = ReshapeLayer(l_dropout1, shape=([0], -1))

    l_encode = DenseLayer(l_reshape1,
                          name='encode',
                          num_units=encode_size)

    l_decode = DenseLayer(l_encode,
                          W=l_encode.W.T,
                          num_units=l_reshape1.output_shape[1])

    l_reshape2 = ReshapeLayer(l_decode,
                              shape=([0],
                                     conv_filters,
                                     int(np.sqrt(l_reshape1.output_shape[1] / conv_filters)),
                                     int(np.sqrt(l_reshape1.output_shape[1] / conv_filters))))

    l_unpool1 = Upscale2DLayer(l_reshape2,
                               scale_factor=pool_size)

    l_de = TransposedConv2DLayer(l_unpool1,
                                 num_filters=l_conv1.input_shape[1],
                                 W=l_conv1.W,
                                 filter_size=l_conv1.filter_size,
                                 stride=l_conv1.stride,
                                 crop=l_conv1.pad,
                                 flip_filters=not l_conv1.flip_filters)

    l_output = ReshapeLayer(l_de,
                            shape=([0], -1))

    return l_output


def build_st_network_MNIST(input_shape,  mins, maxs, ranges, withdisc=True):
    # General Params
    num_filters = 64
    filter_size = (3, 3)
    pool_size = (2, 2)

    # SP Param
    b = np.zeros((2, 3), dtype=theano.config.floatX)
    b[0, 0] = 1
    b[1, 1] = 1
    b = b.flatten()  # identity transform

    # Localization Network
    l_in = InputLayer(shape=(None,
                             input_shape[1],
                             input_shape[2],
                             input_shape[3]))

    l_conv1 = Conv2DLayer(l_in,
                          num_filters=num_filters,
                          filter_size=filter_size)

    l_pool1 = MaxPool2DLayer(l_conv1,
                             pool_size=pool_size)

    l_conv2 = Conv2DLayer(l_pool1,
                          num_filters=num_filters,
                          filter_size=filter_size)

    l_pool2 = MaxPool2DLayer(l_conv2,
                             pool_size=pool_size)

    l_loc = DenseLayer(l_pool2,
                       num_units=64,
                       W=lasagne.init.HeUniform('relu'))

    l_param_reg = DenseLayer(l_loc,
                             num_units=6,
                             b=b,
                             nonlinearity=lasagne.nonlinearities.linear,
                             W=lasagne.init.Constant(0.0),
                             name='param_regressor')

    if withdisc:
        l_dis = DiscreteLayer(l_param_reg, mins, maxs, ranges)
    else:
        l_dis = l_param_reg

    # Transformer Network
    l_trans = TransformerLayer(l_in,
                               l_dis,
                               downsample_factor=1.0)

    # Classification Network
    network = lasagne.layers.Conv2DLayer(
            l_trans, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network
