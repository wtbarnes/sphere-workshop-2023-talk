import pathlib

import astropy.time
from sunpy.coordinates.sun import carrington_rotation_number
from sunpy.net import Fido, attrs as a


ar_date = astropy.time.Time(snakemake.config['ar_date'])
car_rot = carrington_rotation_number(ar_date)
q = Fido.search(
    a.Time('2010/01/01', '2010/01/01'),
    a.jsoc.Series('hmi.synoptic_mr_polfil_720s'),
    a.jsoc.PrimeKey('CAR_ROT', int(car_rot)),
    a.jsoc.Notify(snakemake.config['jsoc_email'])
)
file = Fido.fetch(q, overwrite=True)
pathlib.Path(file[0]).rename(snakemake.output[0])
