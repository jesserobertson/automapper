""" file:   wavelets/utilities.py
    author: Jess Robertson
            CSIRO Mineral Resources
    email:  jesse.robertson@csiro.au
    date:   October 2015

    description: Utility functions
"""

from __future__ import print_function, division

from numpy import broadcast_arrays, einsum, asarray, ndenumerate, \
    sqrt, cos, sin, zeros, identity, pi, log2, arange
from scipy import optimize


def ndmesh(*axes):
    """ Return an n-dimensional mesh generated from a set of basis vectors.

        Behaves the same as numpy.meshgrid but for arbitrary numbers of axes
        rather than just two. We also handle the ordering of axes properly.

        Parameters:
            *axes - a list of arrays giving the samples along each axis

        Returns:
            a grid array, shaped as (N, len(axes[0]), ... len(axes[-1])), where
                N is the number of axes, and then each axis is in the order
                supplied in the aguments
    """
    if len(axes) == 1:
        return asarray(axes[0])
    else:
        result = asarray(broadcast_arrays(
            *[x[(slice(None),) + i * (None,)]
              for i, x in enumerate(map(asarray, axes))]))

        # This reorders the axes so we have (N, x1, ..., xn),
        # not (N, xn, ..., x1)
        trsp_axes = [0] + list(reversed(range(1, result.shape[0] + 1)))
        return result.transpose(*trsp_axes)


def rotate(arr, angles):
    """ Rotate an n-dimensional array through a given set of Euler angles

        Parameters:
            arr - the array to rotate, must be (N, len(x1), ... len(xn))
                shaped, where N is the number of dimensions of the array
            angles - the euler angles for the rotation (see `rotation_matrix`)
                for more details

        Returns:
            an (N, len(x1), ... len(xn)) array containing the rotated data
    """
    # We first transpose ('permute') so that the rotation axis is last,
    # perform the rotation using matrix multiplication, and then transpose
    # back ('demute') so that the array is the right shape
    ndim = len(arr.shape) - 1
    permute = list(range(1, ndim + 1)) + [0]
    demute = [ndim] + list(range(0, ndim))
    return einsum('...j,...ji->...i',
                  arr.transpose(*permute), 
                  rotation_matrix(angles)).transpose(*demute)


def rotation_matrix(angles_list=None, angles_array=None):
    r""" Returns a rotation matrix in n dimensions

        The combined rotation array is build up by left-multiplying the
        preexisting rotation array by the rotation around a given axis.
        For a $d$-dimensional array, this is given by:

        $$ C(\theta) = R(\theta_{d, d-1})R(\theta_{d, d-2})\times\ldots\times
            R(\theta_{i, j})\times\ldots R(\theta_{1, 2}) $$

        where $i$ and $j$ are positive integers ranging from 1 to $d$, and
        satisfy $i \leq j$.

        The rotation matrix is specified using Euler angles - you need
        $d*(d-1)/2$ for a $d$-dimensional array.

        Parameters:
            angles_list - a list of Euler rotation angles for the n-dimensional
                rotation. The function will guess the dimensionality of
                your rotation matrix from the number of angles you supply, and
                will raise a ValueError if there are not d*(d-1) / 2 angles.
            angles_array - a lower-triangular array of Euler rotations (this
                is generated from angles_list, but we provide this argument
                so you can use a pre-computed array if you want).

        Only one of angles_list or angles_array should be specified, and a
        ValueError will be raised if you specify both.

        Returns:
            a rotation matrix for the specified Euler angles.
    """
    # Check inputs
    if angles_list is not None and angles_array is not None:
        raise ValueError('You should only supply one of the angles_list'
                         ' or angles_array arguments to rotation_matrix')

    elif angles_list is not None:
        # Make sure that we have the right number of angles supplied,
        # guess the dimension required
        dimension_estimate = int(1 + sqrt(1 + 8 * len(angles_list))) // 2
        checks = [int(dimension_estimate) - 1, int(dimension_estimate)]
        allowed_angles = set(d * (d - 1) / 2 for d in checks)
        if len(angles_list) not in allowed_angles:
            err_string = (
                'Wrong number of angles ({0}) supplied to rotation_matrix - '
                'you should specify d*(d-1)/2 angles for a d-dimensional '
                'rotation matrix (i.e. {1[0]} angles for d={2[0]} or {1[1]} '
                'angles for d={2[1]})'
            ).format(len(angles_list), checks, allowed_angles)
            raise ValueError(err_string)
        else:
            dim = dimension_estimate

        # Generate angles array from list
        angles_array = zeros((dim, dim))
        angles_gen = (a for a in angles_list)
        for idx, _ in ndenumerate(angles_array):
            if idx[0] > idx[1]:
                angles_array[idx] = next(angles_gen)

    elif angles_array is not None:
        angles_array = asarray(angles_array)
        dim = angles_array.shape[0]

    # Generate rotation matrix
    ident = identity(dim)
    combined = ident.copy()
    for idx, angle in ndenumerate(angles_array):
        # Make sure we're on the lower-diagonal part of the angles array
        if idx[0] <= idx[1]:
            continue

        # Build non-zero elements of rotation matrix using Givens rotations
        # see: https://en.wikipedia.org/wiki/Givens_rotation
        rotation = ident.copy()
        rotation[idx[0], idx[0]] = cos(angle)
        rotation[idx[1], idx[1]] = cos(angle)
        rotation[idx[0], idx[1]] = sin(angle)
        rotation[idx[1], idx[0]] = -sin(angle)

        # Build combined rotation matrix
        combined = combined.dot(rotation)

    return combined


def nyquist_bandwidth(func, bandwidth_guess, threshold=1e-3):
    """ Get the support of a one-dimensional function in Fourier space given a
        guesstimate location of the bandwidth.
        
        Parameters
            func - The function to determine the bandwidth of
            bandwidth_guess - An approximation to the bandwidth of the
                function. Better as an upper bound (i.e. make it too large) as this
                function uses Brent-method root-finding to find the bandwidth.
            threshold - The threshold for determining the location of the
                bandwidth, given as a fraction of the maximum peak of the Fourier
                function. Optional, defaults to 1e-3.
        
        Returns:
            the support of the function
    """
    result = optimize.minimize_scalar(
        lambda x: -abs(func(x)),
        bounds=(0, bandwidth_guess))
    if result:
        maxloc = result['x']
        threshold = -result['fun'] * threshold
        upper_bound = optimize.brentq(
            lambda x: (abs(func(x)) + abs(func(-x))) - threshold,
            maxloc, bandwidth_guess)
    return upper_bound


def generate_scales(n_scales, shape, spacing, minimum_spacing):
    """ Generate a set of wavelet scales given a signal shape 
        and domain spacing in dimension d
    
        Parameters:
            n_scales - number of scales to generate
            shape - the signal shape (d-length tuple)
            spacing - the spacing (d-length tuple)
            minimum_spacing - the minimum spacing required by the
                wavelet at scale 1 (the Nyquist bandwidth times the 
                wavelet scale-wavelength ratio)
        
        Returns:
            a list of scales for the continuous wavelet transform
    """
    # Calculate the maximum frequency sampled by the signal, and use this
    # to calculate the Nyquist scale, which is our minimum
    ndim = len(shape)
    max_frequency = min([
        (2. * pi) / (shape[n] * spacing[n]) * (shape[n] // 2) 
        for n in range(ndim)])
    min_scale = minimum_spacing / max_frequency

    # The maximum scale can be calculated from the size of the domain
    # We'll assume that the edge effects go like scale * srqt(2), so the
    # maximum useful size is something like signal_length / sqrt(2)
    xmax = max(sh * sp / sqrt(2) for sh, sp in zip(shape, spacing))
    scale_spacing = log2(xmax / min_scale) / n_scales
    max_scale = min_scale * 2 ** (n_scales * scale_spacing)

    # Set up the scales and return them
    return min_scale * 2 ** (arange(n_scales) * scale_spacing)