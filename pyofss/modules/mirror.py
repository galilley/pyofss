
"""
    Copyright (C) 2020 Vladislav Efremov
    
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
from pyofss.field import fft, ifft


class Mirror(object):


    def __init__(self, name = 'mirror' , centre_nu = 290.0, slope = 5000.0):

        self.name = name
        self.centre_nu = centre_nu
        self.slope = slope

    def __call__(self, domain, field):

        self.field = fft(field[0])

        factor = 1.0 / ( np.exp( self.slope*(domain.nu/self.centre_nu - 1.0) ) + 1.0 )
        self.field *= factor
        self.field = ifft(self.field)

        return [field[0] - self.field, self.field]


if __name__ == "__main__":
    from pyofss import *

    domain = Domain(centre_nu=193.0, samples_per_bit=1024*16,
                    bit_width = 1600.0, channels = 2)
    ds = Diss_soliton(width = 120., peak_power = 10., C = 600)
    mr = Mirror(centre_nu = domain.centre_nu)
    sys = System( domain )
    sys.add( ds )
    sys.add( mr )
    sys.run()

    double_plot( domain.nu, spectral_power(sys.field[0]),
                 domain.nu, spectral_power(sys.field[1]))
