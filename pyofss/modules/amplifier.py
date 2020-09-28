
"""
    Copyright (C) 2011, 2012  David Bolt

    This file is part of pyofss.

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

from scipy import power, sqrt
from pyofss.field import fft, ifft
from pyofss.field import energy
import numpy as np


class Amplifier(object):
    """
    :param string name: Name of this module
    :param double gain: Amount of (logarithmic) gain. *Unit: dB*
    :param double E_saturation: Energy of saturation gain. *Unit: nJ*
    :param double P_saturation: Power of saturation gain. *Unit: W*
    :param double rep_rate: Repetition rate of pulse. *Unit: MHz*

    Simple amplifier provides gain but no noise
    """
    def __init__(self, name="amplifier", gain=None,
                 E_sat=None, P_sat=None, rep_rate=None):

        if gain is None:
            raise Exception("The gain is not defined.")

        if (P_sat is not None) and (E_sat is not None):
            raise Exception("There should be only one parameter of saturation.")

        self.name = name
        self.gain = gain

        if E_sat is not None:
            self.E_sat = E_sat
        elif P_sat is not None:
            if rep_rate is None:
                raise Exception("Repetition rate is not defined.")
            self.E_sat = 1e3*P_sat/rep_rate   #nJ
        else:
            self.E_sat = None
        self.field = None

    def __call__(self, domain, field):
        # Convert field to spectral domain:
        self.field = fft(field)

        # Calculate linear gain from logarithmic gain (G_dB -> G_linear)
        G = power(10, 0.1 * self.gain)
        if self.E_sat is not None:
            E = energy(field, domain.t)
            G = G/(1.0 + E/self.E_sat)
        sqrt_G = sqrt(G)

        if domain.channels > 1:
            self.field[0] *= sqrt_G
            self.field[1] *= sqrt_G
        else:
            self.field *= sqrt_G

        # convert field back to temporal domain:
        return ifft(self.field)

