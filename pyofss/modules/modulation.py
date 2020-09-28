
"""
    Copyright (C) 2020 Vlad Efremov, Denis Kharenko

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
from pyofss.field import temporal_power


class Modulation(object):
    "SAM"

    def __init__(self, name="modulation",
                 Pcr=None, rho_max=None, rho_min=None):
        self.name = name
        self.Pcr = Pcr
        self.rho_max = rho_max
        self.rho_min = rho_min
        self.g = None

    def __call__(self, domain, field):
        P = temporal_power(field)
        self.g = self.rho_max - (P/self.Pcr - 1.)**2*(self.rho_max - self.rho_min)
        self.g = np.where(self.g < self.rho_min, self.rho_min, self.g)
        return field * self.g
