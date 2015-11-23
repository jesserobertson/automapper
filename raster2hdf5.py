#! /usr/bin/env python3

import os
import numpy as np
import rasterio
import click
import h5py
from affine import Affine
from pyproj import Proj

# raster = os.path.join(config.data_path, 'geophysics/magnetics/'
#                                         'WA_80m_Mag_Merge_1VD_v1_2014.tif')


@click.command()
@click.argument('raster', type=click.Path(exists=True), required=1)
@click.argument('outputdir', type=click.Path(exists=True), required=2)
@click.option('--verbose/--quiet', help="Display conversion status",
              default=True)
def main(raster, outputdir, verbose):
    """ Convert a raster file (e.g. GeoTIFF) into an HDF5 file.

        RASTER: Path to raster file to convert to hdf5.

        The HDF5 file has the following datasets:

            - Raster: (original image data)

            - Latitude: (vector or matrix of pixel latitudes)

            - Longitude: (vector or matrix of pixel longitudes)

        And the following attributes:

            - affine: The affine transformation of the raster

            - Various projection information.
    """

    if verbose:
        print("Opening raster ...")

    # Read raster bands directly to Numpy arrays.
    # Much of this is from:
    #   http://gis.stackexchange.com/questions/129847/
    #   obtain-coordinates-and-corresponding-pixel-values-from-geotiff-using
    #   -python-gdal
    with rasterio.open(os.path.expanduser(raster)) as f:
        T0 = f.affine  # upper-left pixel corner affine transform
        crs = f.crs
        p1 = Proj(crs)
        I = f.read()
        nanvals = f.get_nodatavals()

    # Make sure rasterio is always giving us a 3D array
    assert(I.ndim == 3)

    # This only works on lat-lon projections for now
    if not p1.is_latlong():
        print("Error: This only works on spherical projections for now (YAGNI"
              " you know)...")
        exit(1)

    if verbose:
        print("Extracting coordinate sytem ...")

    # Get affine transform for pixel centres
    T1 = T0 * Affine.translation(0.5, 0.5)

    # Just find lat/lons of axis if there is no rotation/shearing
    # https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations
    if (T1[1] == 0) and (T1[3] == 0):
        lons = T1[2] + np.arange(I.shape[2]) * T1[0]
        lats = T1[5] + np.arange(I.shape[1]) * T1[4]

    # Else, find lat/lons of every pixel!
    else:
        print("Error: Not yet tested... or even implemented properly!")
        exit(1)

        # Need to apply affine transformation to all pixel coords
        cls, rws = np.meshgrid(np.arange(I.shape[2]), np.arange(I.shape[1]))

        # Convert pixel row/column index (from 0) to lat/lon at centre
        rc2ll = lambda r, c: (c, r) * T1

        # All eastings and northings (there a better way to do this)
        lons, lats = np.vectorize(rc2ll, otypes=[np.float, np.float])(rws, cls)

    # Permute layers to be more like a standard image, i.e. (band, lon, lat) ->
    #   (lon, lat, band)
    I = (I.transpose([2, 1, 0]))[:, ::-1]
    lats = lats[::-1]

    # Mask out NaN vals if they exist
    if nanvals is not None:
        for v in nanvals:
            if v is not None:
                if verbose:
                    print("Writing missing values")
                I[I == v] = np.nan

    # Now write the hdf5
    if verbose:
        print("Writing HDF5 file ...")

    file_stump = os.path.basename(raster).split('.')[-2]
    hdf5name = os.path.join(outputdir, file_stump + ".hdf5")
    with h5py.File(hdf5name, 'w') as f:
        drast = f.create_dataset("Raster", I.shape, dtype=I.dtype, data=I)
        drast.attrs['affine'] = T1
        for k, v in crs.items():
            drast.attrs['k'] = v
        f.create_dataset("Latitude", lats.shape, dtype=float, data=lats)
        f.create_dataset("Longitude", lons.shape, dtype=float, data=lons)

    if verbose:
        print("Done!")

if __name__ == '__main__':
    main()
