""" file:   transforms.py
    author: Jess Robertson
            CSIRO Mineral Resources
    email:  jesse.robertson@csiro.au
    date:   October 2015

    description: Functions for handling FFT transforms of signals
"""

from __future__ import print_function, division

from numpy import pi, array, floor, log2, empty
from pyfftw.interfaces.scipy_fftpack import *

from .utilities import ndmesh


def fft_frequencies(signal_shape, sample_spacing):
    """ Calculate the frequencies in Fourier space corresponding to
        the given signal positions

        Parameters:
            shape - A n-dimension length tuple containing the shape
                of the signal
            sample_spacing - The sample spacing along each axis
            angles - Euler angles giving the rotation axis for frequencies
                to be rotated around.

        Returns:
            an n-dimensional array of frequencies
    """
    ndim = len(signal_shape)
    return ndmesh(*[(2. * pi) / (signal_shape[n] * sample_spacing[n])
                    * array(list(range(0, signal_shape[n] // 2))
                            + list(range(-signal_shape[n] // 2, 0)))
                    for n in range(ndim)])


def pad_array(arr):
    """ Pad an n-dimensional array with the mean value so that the lengths of all axes
        are powers of two

        Parameters:
            arr - The array to pad

        Returns:
            A tuple of `(parr, mask)` where `parr` is the padded
            array and `mask` is the subset of `parr` represnting the
            original array. You can get the old array back with
            `parr[mask]`.
    """
    # If we're going to pad the data then find out how many zeros we
    # need to pad
    _mean = arr.mean()
    current_shape = array(arr.shape, dtype=int)
    current_exponent = \
        floor(log2(current_shape)).astype(int)
    nzeros = 2 ** (current_exponent + 1) - current_shape
    nzeros[current_shape == 1] = 0  # Make sure we don't double up
                                    # on axes with one value
    new_shape = current_shape + nzeros

    # Generate padded signal & mask, copy over signal
    padded_array = empty(shape=new_shape, dtype=arr.dtype)
    padded_array.fill(_mean)
    pad_mask = tuple(slice(None, -n or None, None) for n in nzeros)
    padded_array[pad_mask] = arr
    return padded_array, pad_mask


def cwtn(data, scales, angles=None, wavelet=None, pad=True):
    """ N-dimensional continuous wavelet transform 

        Parameters:
            data - an N-dimensional array of data to transform
            scales - the scales to evaluate the transform at
            angles - optional, the angles to evaulate the wavelet at
            wavelet - function to return the FT of a wavelet at the given 
                frequency and scale. Optional, default is derivative of 
                gaussian
            pad - whether to pad the data out to a power of 2 (default True as
                this is a good idea for the FFTs and also to reduce edge
                effects in the transform).

        Returns:
            The wavelet transform
    """
    # Get wavelets
    if wavelet is None:
        wavelet = lambda f, s: dgauss_nd(f, scale=s, order=1)

    # Generate FFT of data
    if pad:
        data, pad_mask = pad_array(data)
    fft_data = fftn(data)

    # Loop through scales and angles
    if angles is not None:
        for angle in angles:
            freq = None
            for scale in scales:
                pass

    else:
        freq = None
        for scale in scales:
            result = ifftn(fft_data * wavelet(freq, scale))[pad_mask]
