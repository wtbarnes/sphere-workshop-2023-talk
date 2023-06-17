import numpy as np
import sunpy.map
from sunpy.coordinates import propagate_with_solar_surface
from sunpy.coordinates.utils import solar_angle_equivalency
import astropy.units as u
from astropy.coordinates import SkyCoord
import pfsspy

import synthesizAR
from synthesizAR.util import change_obstime, change_obstime_frame, from_pfsspy
from mocksipipeline.detector.response import Channel


# Load maps
m_hmi = sunpy.map.Map(snakemake.input[0])
m_aia = sunpy.map.Map(snakemake.input[1])

# Define AR bounding box
# TODO: Should make these corner coordinates a config option
blc_ar = SkyCoord(Tx=-100*u.arcsec, Ty=-525*u.arcsec, frame=m_aia.coordinate_frame)
trc_ar = SkyCoord(Tx=250*u.arcsec, Ty=-325*u.arcsec, frame=m_aia.coordinate_frame)
blc_ar_synop = change_obstime(blc_ar.transform_to(change_obstime_frame(m_hmi.coordinate_frame, blc_ar.obstime)), m_hmi.date)
trc_ar_synop = change_obstime(trc_ar.transform_to(change_obstime_frame(m_hmi.coordinate_frame, trc_ar.obstime)), m_hmi.date)

# Perform field extrapolation
m_hmi_resample = m_hmi.resample((1080,540)*u.pix)
nrho = 70
rss = 2.5
pfss_input = pfsspy.Input(m_hmi_resample, nrho, rss)
pfss_output = pfsspy.pfss(pfss_input)

# Trace fieldlines
num_seeds = 3000
pix_blc = m_hmi_resample.wcs.world_to_pixel(blc_ar_synop)
pix_trc = m_hmi_resample.wcs.world_to_pixel(trc_ar_synop)
pixel_random = np.random.uniform(low=pix_blc, high=pix_trc, size=(num_seeds,2)).T
seeds = m_hmi_resample.wcs.pixel_to_world(*pixel_random).make_3d()
ds = 0.05
max_steps = int(np.ceil(2 * nrho / ds))
tracer = pfsspy.tracing.FortranTracer(step_size=ds, max_steps=max_steps)
fieldlines = tracer.trace(SkyCoord(seeds), pfss_output,)

# Build loops from coordinates
strands = from_pfsspy(fieldlines.closed_field_lines,
                      n_min=100,
                      obstime=m_aia.date,
                      length_min=20*u.Mm,
                      length_max=400*u.Mm,
                      cross_sectional_area=1e16*u.cm**2)
with propagate_with_solar_surface():
    strands = [synthesizAR.Loop(l.name,
                                l.coordinate.transform_to(m_aia.observer_coordinate.frame),
                                field_strength=l.field_strength,
                                cross_sectional_area=l.cross_sectional_area,) for l in strands]
strands_local = []
for s in strands:
    coord = s.coordinate.transform_to(m_aia.coordinate_frame)
    if np.any(coord.Ty> -250*u.arcsec):
        continue
    if np.any(coord.Ty< -600*u.arcsec):
        continue
    if np.any(coord.Tx > 200*u.arcsec):
        continue
    if np.any(coord.Tx < -70*u.arcsec):
        continue
    strands_local.append(s)

# Build skeleton
skeleton = synthesizAR.Skeleton(strands_local)
chan = Channel('filtergram_1')
angular_res = chan.resolution[0]
delta_angle = angular_res / 2 * u.pix
delta_s = delta_angle.to(
    u.Mm,
    equivalencies=solar_angle_equivalency(skeleton.loops[0].coordinate.observer)
)
skeleton_coarse = skeleton.refine_loops(delta_s, prepkwargs={'k': 1})
skeleton_coarse.to_asdf(snakemake.output[0])