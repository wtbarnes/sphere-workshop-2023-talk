"""
MOXSI instrument class for producing DEMs at MOXSI plate scale and resolution
"""
import astropy.units as u
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.wcs.utils import pixel_to_pixel
import dask.array
import ndcube
import numpy as np
import zarr

from mocksipipeline.detector.response import Channel, convolve_with_response
from mocksipipeline.util import read_data_cube
from overlappy.util import strided_array
from synthesizAR.instruments import InstrumentDEM


class InstrumentDEMOXSI(InstrumentDEM):
    name = 'MOXSI_DEM'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for c in self.channels:
            c.psf_width = self.psf_width

    @property
    @u.quantity_input
    def resolution(self) -> u.Unit('arcsec / pix'):
        return self._moxsi_channel.resolution

    @property
    def _moxsi_channel(self):
        # This is just to pull out some useful properties of the instrument for
        # resolution, etc.
        return Channel('filtergram_1')

    @property
    def cadence(self) -> u.s:
        return 1 * u.s

    @property
    def psf_width(self) -> u.pix:
        psf_fwhm = 40 * u.arcsec
        return psf_fwhm * gaussian_fwhm_to_sigma / self.resolution

    @property
    def observatory(self):
        return 'CubIXSS'

    @property
    def telescope(self):
        return 'MOXSI'


def sample_spectral_cube(lam, channel, instr_cube_wcs, observer):
    """
    Sample a Poisson distribution based on counts from spectral cube and map counts to detector pixels.
    """
    samples = dask.array.random.poisson(lam=lam, size=lam.shape).sum(axis=0)
    idx_nonzero = dask.array.where(samples>0)
    idx_nonzero = [i.compute() for i in idx_nonzero]
    weights = samples[samples>0].compute()
    # Map counts to detector coordinates
    overlap_wcs = channel.get_wcs(observer)
    idx_nonzero_overlap = pixel_to_pixel(instr_cube_wcs, overlap_wcs, *idx_nonzero[::-1])
    n_rows = channel.detector_shape[0]
    n_cols = channel.detector_shape[1]
    hist, _, _ = np.histogram2d(idx_nonzero_overlap[1], idx_nonzero_overlap[0],
                                bins=(n_rows, n_cols),
                                range=([-.5, n_rows-.5], [-.5, n_cols-.5]),
                                weights=weights)
    return ndcube.NDCube(strided_array(hist, channel.wavelength.shape[0],),
                         wcs=overlap_wcs,
                         unit='photon')


def instr_cube_to_zarr(time_index, root, spec_cube_dir, channel):
    """
    This saves a timestep of the spectral cube in instrument units for a particular
    spectral order to a single index in a Zarr array so that a Dask array can easily be created from it.
    """
    spec_cube = read_data_cube(spec_cube_dir / f'spec_cube_t{time_index}.fits')
    instr_cube = convolve_with_response(spec_cube, channel, electrons=False)
    root['instr_cube'][time_index, ...] = instr_cube.data
