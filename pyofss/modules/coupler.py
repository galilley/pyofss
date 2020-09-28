# -*- coding: utf-8 -*-
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

from scipy import sqrt
from pyofss.field import fft, ifft

class Coupler(object):
    """
    Symmetric fiber coupler 2*2
    """

    def __init__(self, name='coupler'):
        self.name = name
        self.field = None

    def __call__(self, domain, field):
        self.field = fft(field)
        self.field = ifft(self.matrix_oper(self.field[0], self.field[1]))
        return self.field

    def matrix_oper(self, field_in1, field_in2):
        field_out1 = sqrt(2.0)/2.0*(field_in1 + 1j*field_in2)
        field_out2 = sqrt(2.0)/2.0*(1j*field_in1 + field_in2)
        return [field_out1, field_out2]

