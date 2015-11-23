""" file:   wavelets.py
    author: Jess Robertson
            CSIRO Mineral Resources
    email:  jesse.robertson@csiro.au
    date:   October 2015

    description: Functions for taking wavelet transforms of signals
"""

from . import dgauss, utilities
from .utilities import rotate
from .transforms import fft_frequencies, pad_array
