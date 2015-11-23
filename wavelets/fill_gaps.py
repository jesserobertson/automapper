from __future__ import division, print_function

import numpy
from functools import reduce
import scipy.ndimage as ndimage
from .utilities import ndmesh

def _make_polynomial_basis(locations, order=2):
    """ Make a polynomial basis array from a given list of locations

        Parameters:
            locations - an (n_locations, location_dim) shaped array of
                location vectors (i.e. the first row points to the first location,
                the second to the second location and so on).
            order - the highest polynomial order to use
        
        Returns:
            the basis for a polynomial interpolation
    """
    basis = numpy.vstack([locations.transpose() ** (n + 1)
                          for n in range(order)])
    constants = numpy.ones(shape=locations.shape[0])
    basis = numpy.vstack([constants, basis]).transpose()
    return basis


def fit_nd_polynomial(locations, values, unknown_locations,
                      order=2, return_coeffs=False):
    """ Fit a polynomial to a list of known locations in N dimensions
        using a least-squares method

        :param locations: An `(n_locations, N)` shaped array of
            location vectors (i.e. the first row points to the first location,
            the second to the second location and so on).
        :type locations : `numpy.ndarray`
        :param values: An `(n_locations,)` shaped array containing the known
            values for each location
        :type locations: `numpy.ndarray`
        :param locations: An `(n_unknown_locations, N)` shaped array of
            vectors pointing to the locations to interpolate to
        :type locations : `numpy.ndarray`
        :param order: The highest polynomial order to use
        :type order: int
    """
    # Do least squares using SVD of basis
    basis = _make_polynomial_basis(locations, order=order)
    U, S, V = numpy.linalg.svd(basis, full_matrices=False, compute_uv=True)
    V = V.transpose()
    S_inv = numpy.diag(1 / S)
    beta = reduce(numpy.dot, (V, S_inv, U.transpose(), values))

    # Generate interpolation for unknown values
    unknown_basis = _make_polynomial_basis(unknown_locations, order=order)
    interpolation = numpy.dot(beta, unknown_basis.transpose())
    if return_coeffs:
        return interpolation, beta
    else:
        return interpolation

def fill_gaps(signal, domain=None, border=2, order=1, return_gap_masks=False):
    """ Plug the gaps in an N dimensional signal

        The signal is modified in place and will be returned with all the gaps
        interpolated over. If return_gaps is true, then an `n_gaps` length list
        of masks will also be returned, with each member showing the location
        of a seperate gap in the signal.

        :param signal: An `(nx, ny, nz, ...)`-shaped array containing the
            signal
        :type signal: `numpy.ndarray`
        :param domain: An `(nx, ny, nz..., ndim)` array of vectors specifying
            the domain for the signal (i.e. `domain[idx]` gives the `ndim`
            length location vector of the value in `signal[idx]`). Optional.
        :type domain: `numpy.ndarray`
        :param order: The order of the polynomial used to generate the
            least-squares interpolation used to fill the gaps in the signal.
        :type order: int
    """
    # Generate domain if we don't have one
    if domain is None:
        n_dim = len(signal.shape)
        domain = ndmesh(*([numpy.linspace(0, 1)] * n_dim))
        if n_dim == 1:
            domain = numpy.arange(len(signal))[numpy.newaxis].transpose()
        else:
            domain = ndmesh(*([numpy.arange(len(signal))] * n_dim))

    # Find the gaps
    mask = numpy.isnan(signal)
    labelled_array, nlabels = ndimage.label(mask)
    label_masks = []
    for label in range(1, nlabels + 1):
        # Find the bordering pixels for which we have signal values
        label_mask = (labelled_array == label)
        border_pixels = (
            ndimage.binary_dilation(label_mask,
                                    iterations=border).astype(int)
            - label_mask
        ).astype(bool)
        border_values = signal[border_pixels]
        border_locations = domain[border_pixels]

        # Toss out border values which are also NaN
        nan_mask = nnot(numpy.isnan(border_values))
        border_values = border_values[nan_mask]
        border_locations = border_locations[nan_mask]

        # Generate least-squares fit for each hole and plug the gap
        interpolation = fit_nd_polynomial(
            locations=border_locations,
            values=border_values,
            unknown_locations=domain[label_mask],
            order=order)
        signal[label_mask] = interpolation
        if return_gap_masks:
            label_masks.append(label_mask)

    if return_gap_masks:
        return label_masks
