import os
import sys
import numpy as np
import astropy.units as u

import synthesizAR
from synthesizAR.interfaces.ebtel import EbtelInterface, read_xml

import paths
sys.path.append(paths.scripts.as_posix())
from heating import PoissonHeating


ebtel_config = read_xml(os.path.join(snakemake.config['ebtel_directory'], 'config', 'ebtel.example.cfg.xml'))
simulation_time = float(snakemake.config['simulation_time']) * u.s
ebtel_config['total_time'] = simulation_time.to_value('s')
ebtel_config['use_flux_limiting'] = True
ebtel_config['saturation_limit'] = 1/6
ebtel_config['use_adaptive_solver'] = True
ebtel_config['heating']['background'] = 1e-6
power_law_index = 2
sampling_interval = 10 * u.s
event_duration = 200 * u.s
heating_model = PoissonHeating(simulation_time,
                               float(snakemake.params.frequency),
                               power_law_index,
                               event_duration,
                               interval=sampling_interval)
ebtel_interface = EbtelInterface(ebtel_config, heating_model, snakemake.config['ebtel_directory'])
skeleton = synthesizAR.Skeleton.from_asdf(snakemake.input[0])
skeleton.load_loop_simulations(ebtel_interface, filename=snakemake.output[0])
skeleton.to_asdf(snakemake.output[1])
