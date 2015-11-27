""" file:   dgauss.py
    author: Jess Robertson
            CSIRO Mineral Resources
    email:  jesse.robertson@csiro.au
    date:   October 2015

    description: Derivative-of-Gaussian wavelet functions
"""

from __future__ import print_function, division

import numpy as np
from scipy.special import gamma
from functools import reduce

from . import utilities


def gaussian(freq, scale):
    """ Fourier representation for Gaussian function

        Parameters:
            freq - the frequencies at which to evaluate the transform
            scale - the scale of the wavelet (in spacing units)

        Returns:
            an array containing the fourier transfomr of the wavelet evaluated
                at the given frequencies.
    """
    return np.exp(-((scale * freq) ** 2) / 2.) / np.sqrt(gamma(0.5))


def dgauss(freq, scale, order):
    """ Fourier representation for the Derivative-of-Gaussian wavelet

        Parameters:
            freq - the frequencies at which to evaluate the transform
            scale - the scale of the wavelet (in spacing units)
            order - the order of the derivative. `order=0` corresponds to
                Gaussian smoothing, `order=m` with m > 0 corresponds to a
                smoothed m-th derivative.

        Returns:
            an array containing the fourier transfomr of the wavelet evaluated
                at the given frequencies.
    """
    return (-1j) ** order / np.sqrt(gamma(order + 0.5)) \
           * (scale * freq) ** order \
           * np.exp(-((scale * freq) ** 2) / 2.)


def dgauss_nd(freq, scale, order):
    """ N-dimensional derivative of Gaussian wavelet

        Parameters:
            freq - the frequencies at which to evaluate the transform
            scale - the scale of the wavelet (in spacing units)
            order - the order of the derivative. `order=0` corresponds to
                Gaussian smoothing, `order=m` with m > 0 corresponds to a
                smoothed m-th derivative.

        Returns:
            an array containing the fourier transfomr of the wavelet evaluated
                at the given frequencies.
    """
    idx = range(freq.shape[0] - 1)
    return reduce(lambda a, b: a * b,
                  (gaussian(freq[i], scale=scale) for i in idx),
                  dgauss(freq[-1], scale=scale, order=order))


def dgauss_nd_sym(freq, scale, order):
    """ N-dimensional symmetrical derivative of Gaussian wavelet

        Parameters:
            freq - the frequencies at which to evaluate the transform
            scale - the scale of the wavelet (in spacing units)
            order - the order of the derivative. `order=0` corresponds to
                Gaussian smoothing, `order=m` with m > 0 corresponds to a
                smoothed m-th derivative.

        Returns:
            an array containing the fourier transfomr of the wavelet evaluated
                at the given frequencies.
    """
    sym_freq = np.sqrt(np.sum(f ** 2 for f in freq))
    return 1 / np.sqrt(gamma(order + 0.5)) \
           * (scale * sym_freq) ** order \
           * np.exp(-((scale * sym_freq) ** 2) / 2.)


def scale_wavelength_ratio(order):
    """ Convert DOG scale parameter to equivalent Fourier wavelength
    """
    return 2 * np.pi / np.sqrt(order + 0.5)


def generate_scales(n_scales, shape, spacing, order):
    """ Generate scales for the Derivative-of-Gaussian wavelet

        This does the same thing as the utilities.generate_scales 
        but handles the issues around minimum sample spacing/
        Nyquist frequencies for you

        Parameters:
            n_scales - the number of scales to generate
            shape - the shape of the signal (an n-length tuple for
                a n-dimensional signal)
            spacing - the sample spacing for the signal domain
            order - the order of the Derivative-of-Gaussian wavelet

        Returns:
            an array of scales logarithmically distributed from 
            smallest to largest
    """
    wav = lambda f: dgauss(f, scale=1, order=order)
    minimum_spacing = utilities.nyquist_bandwidth(wav, 10) \
        * scale_wavelength_ratio(order=order)
    return utilities.generate_scales(n_scales, shape, spacing, 
                                     minimum_spacing)
