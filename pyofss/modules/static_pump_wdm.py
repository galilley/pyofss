
"""
    Copyright (C) 2020 Vladislav Efremov, Denis Kharenko

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

import numpy as np

from pyofss.field import fft, ifft, ifftshift, fftshift
from pyofss.domain import lambda_to_nu

# Define exceptions
class StaticPumpWDMError(Exception):
    pass

class DiffArraysError(StaticPumpWDMError):
    pass

class StaticPumpWDM(object):
    """
    The module adds predefined (static) field (pump) to the calculated field in WDM manner.
    It replaces items of the field array above (lowpass) or below (highpass) cutoff frequency.

    :param string name: Name of this module
    :param string type: specify which radiation will be past and used next (lowpass or highpass)
    :param double cutoff_nu: Cutoff frequency relative to domain centre
                           frequency at which the zeroing and replacement
                           begins. *Unit: THz*
    :param array field_repl: An array of fields that will be replaced

    We use high-pass and low-pass notation implying frequencies rather that wavelengths
    """

    def __init__(self, name = 'wdm', type_wdm = "lowpass",
                 cutoff_nu = None, field_pump = None):

        self.name = name
        self.cutoff_nu = cutoff_nu
        self.field_pump = field_pump
        self.ind = None
        self.field = None
        if type_wdm == "lowpass":
            self.dir_to_up = False
        else:
            self.dir_to_up = True

    def __call__(self, domain, field):

        if self.ind is None:
            self.ind = max(np.where(domain.nu < self.cutoff_nu)[0])

            if self.field_pump is not None:
                self.field_pump = ifftshift(fft(self.field_pump))
                if self.dir_to_up is True:
                    self.field_pump[self.ind:] = 0.0
                else:
                    self.field_pump[:self.ind] = 0.0
            else:
                self.field_pump = np.zeros(field.size, dtype=field.dtype)

        if not (len(self.field_pump) == len(field)):
            raise DiffArraysError(
                "arrays of fields is different lengths")

        self.field = ifftshift(fft(field))

        if self.dir_to_up is True:
            self.field[:self.ind] = 0.0
        else:
            self.field[self.ind:] = 0.0

        return ifft(fftshift(self.field + self.field_pump))


if __name__ == "__main__":
    from pyofss import *
    cwl = 1030.0
    domain = Domain(
            centre_nu=lambda_to_nu(cwl),
            samples_per_bit=1024*16,
            bit_width = 1600.0)
    ds = Diss_soliton(
            width = 30.,
            peak_power = 10.,
            offset_nu = dlambda_to_dnu(5, cwl),
            C = 60)
    sech = Sech(
            width = 10,
            peak_power = 5.0,
            offset_nu = -dlambda_to_dnu(10, cwl))

    sys = System( domain )
    sys.add( sech )
    sys.run()
    field_pump = np.copy(sys.field)
    sys.clear()

    wdm = StaticPumpWDM(type_wdm = "highpass", cutoff_nu = lambda_to_nu(1035.0), field_pump = field_pump)

    sys = System( domain )
    sys.add( ds )
    sys.add( wdm )
    sys.run()

    double_plot( sys.domain.t, temporal_power( sys.field ),
            sys.domain.Lambda, (spectral_power( sys.field )),
            labels["t"], labels["P_t"], labels["Lambda"], labels["P_lambda"],
            inst_freq = inst_freq(sys.field, sys.domain.dt), y2_label = labels["inst_nu"])
