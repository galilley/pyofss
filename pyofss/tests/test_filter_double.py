
"""
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

from pyofss.modules.filter_doubl import Filter_doubl, OutOfRangeError, NotIntegerError

import unittest2


class DefaultParameters(unittest2.TestCase):
    """ Test default parameters. """
    def test_none(self):
        """ Should use default value if no parameter given """
        gfilter = Filter_doubl()
        self.assertEqual(gfilter.name, "filter_double")
        self.assertEqual(gfilter.type, "reflected")
        self.assertEqual(gfilter.width_nu1, 0.1)
        self.assertEqual(gfilter.width_nu2, 1.0)
        self.assertEqual(gfilter.offset_nu, 0.0)
        self.assertEqual(gfilter.m, 1)
        self.assertEqual(gfilter.channel, 0)
        self.assertIsNone(gfilter.fwhm_nu)
        self.assertEqual(gfilter.a1, 0.3333333333333333)
        self.assertEqual(gfilter.a2, 0.6666666666666666)


class BadParameters(unittest2.TestCase):
    """ Test response to bad parameters. """
    def test_too_low(self):
        """ Should fail when parameters are too low """
        self.assertRaises(OutOfRangeError, Filter_doubl, width_nu1=1e-6)
        self.assertRaises(OutOfRangeError, Filter_doubl, width_nu2=1e-6)
        self.assertRaises(OutOfRangeError, Filter_doubl, offset_nu=-200.0)
        self.assertRaises(OutOfRangeError, Filter_doubl, m=0)
        self.assertRaises(OutOfRangeError, Filter_doubl, channel=-1)
        self.assertRaises(OutOfRangeError, Filter_doubl, relation= -2)

    def test_too_high(self):
        """ Should fail when parameters are too high """
        self.assertRaises(OutOfRangeError, Filter_doubl, width_nu1=1e3)
        self.assertRaises(OutOfRangeError, Filter_doubl, width_nu2=1e3)
        self.assertRaises(OutOfRangeError, Filter_doubl, offset_nu=200.0)
        self.assertRaises(OutOfRangeError, Filter_doubl, m=50)
        self.assertRaises(OutOfRangeError, Filter_doubl, channel=2)

    def test_wrong_type(self):
        """ Should fail if wrong type """
        self.assertRaises(NotIntegerError, Filter_doubl, m=1.4)
        self.assertRaises(NotIntegerError, Filter_doubl, channel=0.5)


class CheckInputAndOutput(unittest2.TestCase):
    """ Test input and output of Filter Double. """
    def test_output(self):
        """ Check filter outputs its values correctly """
        gfilter = Filter_doubl("filter_double", 0.2, 2.0, 10.0, 0.5, 4, 1)
        expected_string = [
                'amplitude1 = 0.090909', 'amplitude2 = 0.909091',
                'width_nu1 = 0.200000 THz', 'width_nu2 = 2.000000 THz', 
                'offset_nu = 0.500000 THz', 'm = 4', 'channel = 1',
                'type_filt = reflected']
        self.assertEqual(str(gfilter), '\n'.join(expected_string))

if __name__ == "__main__":
    unittest2.main()
