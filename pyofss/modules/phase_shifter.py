"""
    Copyright (C) 2023 Vladislav Efremov, Denis Kharenko

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


class Phase_shifter(object):
    def __init__(self, name='ph_shifter', channel=0,
                 phase=None):
        self.name = name
        self.channel = channel
        self.phase = phase

        self.field = None

    def __call__(self, domain, field):
        self.field = field.copy()
        factor = self.phase*1j
        if domain.channels > 0:
            self.field[self.channel] = np.exp(factor) * (self.field[self.channel])
        else:
            self.field = np.exp(factor) * self.field
        return self.field

