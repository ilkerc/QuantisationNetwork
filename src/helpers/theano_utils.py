import theano
import numpy as np
import theano.tensor as T


def ssim(inputs, targets):
    """
    This function calculates the structural similarity btw inputs and targets

    :param inputs: inputs are assumed to be 2D array (input_size, linearFeatures)
    :param targets: targets are assumed to be 2D array (input_size, linearFeatures)
    :return: inverse ssim metric, 1-ssim indicates the cost
    """
    outputs, updates = theano.scan(fn=theano_ssim,
                                   sequences=[inputs, targets],
                                   n_steps=inputs.shape[0])
    return outputs.mean()


def theano_ssim(img1, img2):
    s1 = img1.sum()
    s2 = img2.sum()
    ss = (img1 * img1).sum() + (img2 * img2).sum()
    s12 = (img1 * img2).sum()
    vari = ss - s1 * s1 - s2 * s2
    covar = s12 - s1 * s2
    ssim_c1 = .01 * .01
    ssim_c2 = .03 * .03
    ssim_value = (2 * s1 * s2 + ssim_c1) * (2 * covar + ssim_c2) / ((s1 * s1 + s2 * s2 + ssim_c1) * (vari + ssim_c2))
    return 1 - ssim_value


# Reconstruction error
def compute_reconstruction_error(inputs, targets, outputs):
    """
    This function computes the reconstruction error.
    Metric 1 : Normalized rms structural transformation error : Each inputs distance to the target
    (targets are assumed to be fixed) will be calculated & summed. The final value is the mean of this summation
    This metric is normalized over target image

    Metric 2 : Normalized rms Regression error : Each outputs distance to the target will be calculated & summed.
    The final value is the mean of this summation.

    :param inputs: The augmented samples 4D Tensor
    :param targets: The target (original image) 4D Tensor, the size equals to inputs, but all the same
    :param outputs: The output of the network (regression result) 4D tensort
    :return: reconstruction error
    """
    # Make input also linear so element wise division can be performed
    inputs_res = T.reshape(inputs, (inputs.shape[0], -1))

    # Metric 1
    mu_m1 = T.sum((inputs_res - targets) ** 2)
    mu_m1 /= targets.sum(axis=None) ** 2
    mu_m1 = T.sqrt(mu_m1)

    # Metric 2
    mu_m2 = T.sum((outputs - targets) ** 2)
    mu_m2 /= targets.sum(axis=None) ** 2
    mu_m2 = T.sqrt(mu_m2)

    # Ratio of two metrics, highest expectation is mu_m1
    return mu_m2 / mu_m1
