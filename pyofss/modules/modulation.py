
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
    """
    The module realize a parabolic point action self-amplitude modulation
    (SAM) transmission function expressed as

        rho(P) = rho_max - (P/P_cr - 1)^2 * (rho_max - rho_min),

    which is a quite good approximation for NPE-based artifisial saturable 
    absorber [1,2].

    param string name: Name of this module
    param double Pcr: Critical power [W]
    param double rho_max: Maximal transmission coefficient [a.u.] 
                            (rho_min < rho_max < 1)
    param double rho_min: Minimal transmission at P=0 [a.u.]
                            (0 < rho_min < rho_max)

    [1] A. E. Bednyakova et al., Opt. Express 21(18), 20556 (2013).
    [2] A. E. Bednyakova et al., JOSA B 37(9), 2763 (2020).

    """

    def __init__(self, name="modulation",
                 Pcr=None, rho_max=None, rho_min=None):
        self.name = name
        self.Pcr = Pcr
        self.rho_max = rho_max
        self.rho_min = rho_min
        self.g = None

    def __call__(self, domain, field):
        P = temporal_power(field)
        self.g = self.rho_max - ((P/self.Pcr - 1.)**2) * (self.rho_max - self.rho_min)
        self.g = np.where(self.g < self.rho_min, self.rho_min, self.g)
        return field * self.g
