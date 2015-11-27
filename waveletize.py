#!/usr/bin/env python
""" Generate wavelet transforms from HDF5 data
"""

from __future__ import print_function, division

from wavelets import dgauss, pad_array, fft_frequencies, rotate

import h5py, os
from pyfftw.interfaces.scipy_fftpack import fftn, ifftn
import pyprind
from numpy import pi, linspace, float64, nanmean, isnan
import click

@click.command()
@click.argument('filename', type=click.Path(exists=True), required=1)
@click.option('--order', type=int, default=1,
              help="The order of the derivative to calculate")
@click.option('--nangles', type=int, default=1,
              help="The number of angles to calculate transforms for")
@click.option('--nscales', type=int, default=20,
              help="The number of scales to resolve in the transform")
@click.option('--sym', is_flag=True, default=False,
              help="Whether to use a symmetric or asymmetric wavelet"
                   " (default: asymmetric)")
def make_wavelets(filename, order, nangles, nscales, sym):
    """ Make a wavelet transform from an HDF5 file
    """
    # Set wavelet type
    if sym:
        wav = dgauss.dgauss_nd_sym
    else:
        wav = dgauss.dgauss_nd

    # Get info from input signal
    with h5py.File(filename) as src:
        spacing = (
            abs(src['Longitude'][1] - src['Longitude'][0]),
            abs(src['Latitude'][1] - src['Latitude'][0]))
        nxs, nys, _ = src['Raster'].shape
        shape = (nxs, nys)

        # Generate axes for transform
        scales = dgauss.generate_scales(nscales, shape, spacing, order)
        angles = linspace(
            0, pi * (1 - 1 / nangles), nangles)
        axes = [
            (0, 'Angle', angles),
            (1, 'Scale', scales),
            (2, 'Longitude', src['Longitude'][...]),
            (3, 'Latitude', src['Latitude'][...]),
        ]

        # Remove NaNs and pad array...
        raster = src['Raster'][..., 0]
        mean = nanmean(nanmean(raster))
        raster[isnan(raster)] = mean
        pad_raster, pad_mask = pad_array(raster)
        pad_shape = pad_raster.shape
        fft_data = fftn(pad_raster)

        # Generate sink file
        sink_fname = os.path.splitext(filename)[0] \
                     + '_deriv_order{0}.hdf5'.format(order)
        with h5py.File(sink_fname) as sink:
            sink_shape = angles.shape + scales.shape + shape
            sink.require_dataset('Raster', shape=sink_shape, dtype=float64)

            # Attach dimension labels to raster, write to sink
            for idx, label, dim in axes:
                sink.require_dataset(name=label,
                                     shape=dim.shape,
                                     dtype=float64,
                                     exact=True,
                                     data=dim)
                sink['Raster'].dims.create_scale(dset=sink[label], name=label)
                sink['Raster'].dims[idx].attach_scale(sink[label])

            # Evaluate transforms
            progbar = pyprind.ProgBar(len(angles) * len(scales) + 1)
            freqs = fft_frequencies(pad_shape, spacing)
            for aidx, angle in enumerate(angles):
                rfreqs = rotate(freqs, (angle,))
                for sidx, scale in enumerate(scales):
                    item = 'Angle: {0:0.2f} deg, Scale: {1:0.2f} deg'.format(
                        angle * 180 / pi, scale)
                    progbar.update(item_id=item)
                    filtered = ifftn(
                        fft_data * wav(rfreqs, order=order, scale=scale))
                    sink['Raster'][aidx, sidx, ...] = filtered[pad_mask].real

if __name__ == '__main__':
    make_wavelets()
