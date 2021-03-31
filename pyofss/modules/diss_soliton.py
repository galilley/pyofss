
"""
    Copyright (C) 2020 Vlad Efremov

    This file is part of pyofss 2.0.


    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from numpy import pi
import numpy as np
from scipy import sqrt, exp
from pyofss import Sech

class Diss_soliton(Sech):
    """
    :param string name: Name of this module
    :param double position: Relative position within time window
    :param double width: Half width at 1/e of maximum. *Unit: ps*
    :param double peak_power: Power at maximum. *Unit: W*
    :param double offset_nu: Offset frequency relative to domain centre
                            frequency. *Unit: THz*
    :param Uint m: Order parameter. m = 0. **Unused**
    :param double C: Chirp parameter. *Unit: rad*
    :param double initial_phase: Control the initial phase. *Unit: rad*
    :param Uint channel: Channel of the field array to modify if multi-channel.
    :param bool using_fwhm: Determines whether the width parameter is a
                            full-width at half-maximum measure (FWHM), or
                            a half-width at 1/e-maximum measure (HWIeM)

    Generates a pulse with a Dissipative Soliton profile. A HWIeM pulse width is used
    internally; a FWHM pulse width will be converted on initialisation.
    """

    def __init__ (self, name="diss_soliton", position=0.0, width=10.0,
                 peak_power=1e-3, offset_nu=0.0, m=0, C=5.0,
                 initial_phase=0.0, channel=0, using_fwhm=False):
        super(Diss_soliton, self).__init__(name, position, width,
             peak_power, offset_nu, m, C,
             initial_phase, channel, using_fwhm)

    def __call__(self, domain, field):
        """
        :param object domain: A domain
        :param object field: Current field
        :return: Field after modification by Dissipative Solitons
        :rtype: Object

        Source: DOI 10.3367/UFNe.2015.12.037674
        """
        self.field = field

        t_normalised = \
            (domain.t - self.position * domain.window_t) / self.width

        phase = self.initial_phase
        phase -= 2.0 * pi * self.offset_nu * domain.t

        sechh = 1./np.cosh(t_normalised)
        sechh = np.where(sechh != 0, np.power(sechh, 1+1j*self.C), 0.)
        magnitude = sqrt(self.peak_power)*sechh

        if domain.channels > 1:
            self.field[self.channel] += magnitude * exp(1j * phase)
        else:
            self.field += magnitude * exp(1j * phase)

        return self.field

if __name__ == "__main__":
    """ Plot a default Diss_soliton in temporal and spectral domain """
    from pyofss import Domain, System, Diss_soliton
    from pyofss import temporal_power, spectral_power, inst_freq
    from pyofss import double_plot, labels

    sys = System(Domain(bit_width=500.0))
    sys.add(Diss_soliton())
    sys.run()

    double_plot(sys.domain.t, temporal_power(sys.field),
                sys.domain.nu, spectral_power(sys.field, True),
                labels["t"], labels["P_t"], labels["nu"], labels["P_nu"],
                inst_freq = inst_freq(sys.field, sys.domain.dt), y2_label=labels["inst_nu"])

