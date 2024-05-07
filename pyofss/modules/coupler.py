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
    Fiber coupler 2*2
    """

    def __init__(self, ratio = [50, 50], name='coupler', invers = False):
        self.name = name
        self.field = None
        self.invers = invers

        if sum(ratio) != 100:
            raise Exception("The sum from the outputs is not equal to 100 %.")
        self.k = np.arctan( np.sqrt( ratio[1]/ratio[0] ) )


    def __call__(self, domain, field):
        self.field = fft(field)
        if self.invers is False:
            self.field = ifft(self.matrix_oper(self.field[0], self.field[1]))
        else:
            self.field = ifft(self.matrix_oper_invers(self.field[0], self.field[1]))
        return self.field

    def matrix_oper(self, field_in1, field_in2):
        field_out1 = field_in1*np.cos(self.k) + 1j*field_in2*np.sin(self.k)
        field_out2 = 1j*field_in1*np.sin(self.k) + field_in2*np.cos(self.k)
        return np.array([field_out1, field_out2])

    def matrix_oper_invers(self, field_in1, field_in2):
        field_out1 = field_in1*np.cos(self.k) - 1j*field_in2*np.sin(self.k)
        field_out2 = -1j*field_in1*np.sin(self.k) + field_in2*np.cos(self.k)
        return np.array([field_out1, field_out2])

