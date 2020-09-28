
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

class Splitter( object ):

    def __init__(self, name="splitter", loss = 0.0):
        # TODO negative values should be represent as dB
        self.name = name
        self.loss = loss

    def __call__(self, domain, field):
        return np.sqrt(1.0 - self.loss)*field

