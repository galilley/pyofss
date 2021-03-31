
"""
    Copyright (C) 2021 Denis Kharenko

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

from .linearity import Linearity

class Dispersion(object):

    def __init__(self, name = 'disp', beta=None, centre_omega=None):
        """
        :param array_like beta: Array of dispersion parameters

        Simply module provides dispersion shift
        """
        self.name = name
        self.linearity = Linearity(
                beta=beta, centre_omega=centre_omega)

    def __call__(self, domain, field):
        factor = self.linearity(domain)
        return ifft( np.exp(factor) * fft(field) )

