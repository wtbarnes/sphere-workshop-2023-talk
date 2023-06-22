import pathlib

import astropy.time
import astropy.units as u
from sunpy.net import Fido, attrs as a


ar_date = astropy.time.Time(snakemake.config['ar_date'])

q = Fido.search(
    a.Time(ar_date - 12*u.s, ar_date + 12*u.s, near=ar_date),
    a.Instrument.aia,
    a.Wavelength(171*u.angstrom),
)
file = Fido.fetch(q, overwrite=True)
pathlib.Path(file[0]).rename(snakemake.output[1])
