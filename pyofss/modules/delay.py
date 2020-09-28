
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

import numpy as np
from pyofss.field import fft, ifft, fftshift

class Delay(object):

    def __init__(self, name = 'delay', time = 0.0):
        """
        param time: the delay value

        Simply module provides delay in ps
        """
        self.name = name
        self.time = time

    def __call__(self, domain, field):

        self.Domega = domain.omega - domain.centre_omega
        self.factor = 1j*fftshift( self.time*self.Domega )

        return ifft( np.exp(self.factor) * fft(field) )

