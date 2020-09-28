
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

class Sat_absorber(object):
    "saturable absorber"

    def __init__(self, deep, power_sat):
        self.deep = deep
        self.power_sat = power_sat

    def __call__(self, domain, field):
        q = self.deep/(1. + abs(field)**2/self.power_sat)
        return field*(1. - q)
