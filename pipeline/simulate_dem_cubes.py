import pathlib
import sys

import astropy.units as u
import distributed
import numpy as np
from sunpy.coordinates import get_earth
import sunpy.io._fits as sunpy_fits
import sunpy.map

import synthesizAR
from mocksipipeline.physics.spectral import get_spectral_tables

import paths
sys.path.append(paths.scripts.as_posix())
from instruments import InstrumentDEMOXSI


skeleton = synthesizAR.Skeleton.from_asdf(snakemake.input[0])

# Set up instrument
earth_observer = get_earth(skeleton.loops[0].coordinate.obstime)
observing_midpoint = float(snakemake.config['simulation_time']) / 2 * u.s
observing_duration = float(snakemake.config['observation_duration']) * u.s
observing_interval = observing_midpoint + observing_duration * [-.5, .5]
temperature_bin_edges = 10**np.arange(float(snakemake.config['log_t_min']),
                                      float(snakemake.config['log_t_max'])+float(snakemake.config['delta_log_t']),
                                      float(snakemake.config['delta_log_t'])) * u.K
pad_fov = [float(snakemake.config['pad_fov_x']), float(snakemake.config['pad_fov_y'])] * u.arcsec
dem_moxsi = InstrumentDEMOXSI(observing_interval,
                              earth_observer,
                              temperature_bin_edges=temperature_bin_edges,
                              pad_fov=pad_fov)

# Connect to Dask client
client = distributed.Client(address=snakemake.config['client_address'])

# Build DEM maps
dem_output_dir = pathlib.Path(snakemake.output[0])
dem_output_dir.mkdir(parents=True, exist_ok=True)
dem_maps = dem_moxsi.observe(skeleton, save_directory=dem_output_dir, save_kernels_to_disk=True)

# Build spectral cubes
spectra_output_dir = pathlib.Path(snakemake.output[1])
spectra_output_dir.mkdir(parents=True, exist_ok=True)
spec_tables = get_spectral_tables()
spec_table = spec_tables[snakemake.config['spectral_table']][:,:2000]  # cut spectra to wavelengths that fall on detector
for i,t in enumerate(dem_moxsi.observing_time):
    dem_maps = sunpy.map.Map(sorted(list(dem_output_dir.glob(f'*_t{i}.fits'))))
    dem_cube = dem_moxsi.dem_maps_list_to_cube(dem_maps, dem_moxsi.temperature_bin_centers)
    header = dem_maps[0].wcs.to_header()
    for k,v in dem_maps[0].meta.items():
        if k.upper() not in header.keys() and k != 'keycomments':
            header[k] = v
    spec_cube = dem_moxsi.calculate_intensity(dem_cube, spec_table, header)
    sunpy_fits.write(spectra_output_dir / f'spec_cube_t{i}.fits',
                     spec_cube.data,
                     spec_cube.meta,
                     overwrite=True)
