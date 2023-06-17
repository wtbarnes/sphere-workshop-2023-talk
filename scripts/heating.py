"""
Heating model of EBTEL hydrodynamics
"""
import astropy.units as u
import numpy as np

from synthesizAR.models.heating import b_over_l_scaling


class PoissonHeating:
    """
    Implement time-dependent heating approach of Warren et al. (2020)
    """
    
    @u.quantity_input
    def __init__(self, total_time: u.s, p, alpha, duration: u.s, interval=1*u.s):
        """
        total_time
            Total simulation time
        p
            Finite probabiltiy of an event occuring per unit time
        alpha
            Power-law index of the event energy distribution
        duration
            Duration of each heating event
        interval
            Interval to sample when sampling the Poisson distribution of events
        """
        self.total_time = total_time
        self.p = p
        self.alpha = alpha
        self.duration = duration
        self.interval = interval
    
    def get_steady_heating_rate(self, loop)->u.Unit('erg cm-3 s-1'):
        # These parameters will give approximately a DEM centered on 3-4 MK
        return b_over_l_scaling(loop, H_0=8.77e-3*u.Unit('erg cm-3 s-1'),
                                B_0=76*u.G,
                                L_0=29*u.Mm,
                                alpha=0.2,
                                beta=1.0)
    
    def calculate_event_properties(self, loop):
        n_time = int(np.ceil((self.total_time / self.interval).decompose()))
        num_events = np.random.poisson(lam=self.p, size=n_time)
        event_happened = np.where(num_events>0)[0]
        waiting_times = np.diff(np.append(-1, event_happened)) * self.interval
        # We multiply by the number of events and assume that events which happen within a given
        # interval are of the same magnitude and thus add. To avoid this, you can shorten the interval
        # such that the number of events per interval is always 0 or 1. 
        heating_rates = np.exp(self.p * (waiting_times / self.interval).decompose() / self.alpha)
        # Normalize the heating rate
        steady_heating_rate = self.get_steady_heating_rate(loop)
        norm = steady_heating_rate * self.total_time / (self.duration/2*heating_rates.sum())
        heating_rates *= norm
        # Set up event times
        start_rise_times = event_happened * self.interval
        end_rise_times = start_rise_times + self.duration/2
        start_decay_times = start_rise_times + self.duration/2
        end_decay_times = start_rise_times + self.duration
        return {
            'magnitude': heating_rates.to_value('erg cm-3 s-1'),
            'rise_start': start_rise_times.to_value('s'),
            'rise_end': end_rise_times.to_value('s'),
            'decay_start': start_decay_times.to_value('s'),
            'decay_end': end_decay_times.to_value('s'),
        }
