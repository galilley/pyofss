
"""
    Copyright (C) 2021 Vladislav Efremov

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


class Amplifier_ch(object):
    """
    :param string name: Name of this module
    :param double gain: Amount of (logarithmic) gain. *Unit: dB*
    :param double P_saturation: Power of saturation gain. *Unit: W*
    :param double E_saturation: Energy of saturation gain. *Unit: nJ*
    :param double rep_rate: Repetition rate of pulse. *Unit: MHz*

    Simple amplifier provides gain on two channels are independent
    """
    def __init__(self, name="amplifier_ch",
                 gain=None, E_sat=None, P_sat=None, rep_rate=None):

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
        self.E_ch1 = None
        self.E_ch2 = None
        self.sqrt_G = None

    def __call__(self, domain, field):
        #Calculate energy of field for channel I
        self.E_ch1 = energy(field[0], domain.t)
        #Calculate energy of field for channel II
        self.E_ch2 = energy(field[1], domain.t)

        # Calculate linear gain from logarithmic gain (G_dB -> G_linear)
        G = power(10, 0.1 * self.gain)
        if self.E_sat is not None:
            G = G/(1.0 + (self.E_ch1 + self.E_ch2)/self.E_sat)
        self.sqrt_G = sqrt(G)

        return field

class Amplifier_ch_numb(object):
    """
    :param string name: Name of this module
    :param int numb_ch: Number of channel
    :param class ampl_main: Main amplifier, where the gain parametrs come from

    Providing gain field of channels.
    """
    def __init__(self, name="amplifier_ch", numb_ch = 1, ampl_main = None):
        if not 0 < numb_ch < 3:
            raise Exception("Number of channel must be 1 or 2")

        self.name = name + str(numb_ch)
        self.numb_ch = numb_ch
        self.ampl_main = ampl_main

        self.field = None

    def __call__(self, domain, field):
        # Convert field to spectral domain:
        self.field = fft(field[self.numb_ch - 1])

        self.field *= self.ampl_main.sqrt_G
        if self.numb_ch > 1:
            return [field[0], ifft(self.field)]
        else:
            return [ifft(self.field), field[1]]

