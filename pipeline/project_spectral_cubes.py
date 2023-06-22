import sys
import pathlib
import tempfile

from astropy.wcs.utils import wcs_to_celestial_frame
import dask.array
import distributed
import zarr

from mocksipipeline.util import read_data_cube
from mocksipipeline.detector.response import convolve_with_response, SpectrogramChannel
from overlappy.io import write_overlappogram

import paths
sys.path.append(paths.scripts.as_posix())
from instruments import instr_cube_to_zarr, sample_spectral_cube


client = distributed.Client(address=snakemake.config['client_address'])
channel = SpectrogramChannel(int(snakemake.params.spectral_order), full_detector=False)

# Get list of spectral cube files
spectra_output_dir = pathlib.Path(snakemake.input[0])
spec_cube_files = list(spectra_output_dir.glob('spec_cube_t*.fits'))
n_time = len(spec_cube_files)

# NOTE: precomputing one instrument cube here to get the WCS and needed dimensions
tmp_spec_cube = read_data_cube(spec_cube_files[0])
tmp_instr_cube = convolve_with_response(tmp_spec_cube, channel, electrons=False,)
instr_cube_wcs = tmp_instr_cube.wcs
shape = (n_time,) + tmp_instr_cube.data.shape

with tempfile.TemporaryDirectory() as tmpdir:
    # Setup Zarr dataset for storing instrument datacubes
    root = zarr.open(tmpdir, mode='a')
    ds = root.create_dataset(f'instr_cube',
                             shape=shape,
                             chunks=(1,)+tmp_instr_cube.data.shape,
                             overwrite=True)

    # Compute instrument cube for each timestep
    # NOTE: for some reason, I am having a difficult time parallelizing this
    # which is why it is done in a for loop
    for i in range(n_time):
        instr_cube_to_zarr(i, root, spectra_output_dir, channel)
    lam = dask.array.from_zarr(root[f'instr_cube'])

    # Map sampled photons to detector
    pathlib.Path(snakemake.output[0]).parent.mkdir(parents=True, exist_ok=True)
    observer = wcs_to_celestial_frame(tmp_spec_cube.wcs).observer
    overlappogram = sample_spectral_cube(lam, channel, instr_cube_wcs, observer)

write_overlappogram(overlappogram, snakemake.output[0])
