
"""
    This file is part of pyofss.

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

from pyofss.modules.diss_soliton import Diss_soliton
from pyofss.modules.sech import OutOfRangeError, NotIntegerError

from numpy.testing.utils import assert_almost_equal  # uses decimal places

import unittest2


class DefaultParameters(unittest2.TestCase):
    """ Test default parameters. """
    def test_none(self):
        """ Should use default value if no parameter given """
        dsoliton = Diss_soliton()
        self.assertEqual(dsoliton.name, "diss_soliton")
        self.assertEqual(dsoliton.position, 0.5)
        self.assertEqual(dsoliton.width, 10.0)
        self.assertEqual(dsoliton.peak_power, 1e-3)
        self.assertEqual(dsoliton.offset_nu, 0.0)
        self.assertEqual(dsoliton.C, 5.0)
        self.assertEqual(dsoliton.initial_phase, 0.0)
        self.assertEqual(dsoliton.channel, 0)
        self.assertIsNone(dsoliton.fwhm)


class BadParameters(unittest2.TestCase):
    """ Test response to bad parameters. """
    def test_too_low(self):
        """ Should fail when parameters are too low """
        self.assertRaises(OutOfRangeError, Diss_soliton, position=-0.1)
        self.assertRaises(OutOfRangeError, Diss_soliton, width=1e-3)
        self.assertRaises(OutOfRangeError, Diss_soliton, peak_power=-1e-9)
        self.assertRaises(OutOfRangeError, Diss_soliton, offset_nu=-200.0)
        self.assertRaises(OutOfRangeError, Diss_soliton, C=-1e3)
        self.assertRaises(OutOfRangeError, Diss_soliton, initial_phase=-1.0)
        self.assertRaises(OutOfRangeError, Diss_soliton, channel=-1)

    def test_too_high(self):
        """ Should fail when parameters are too high """
        from numpy import pi

        self.assertRaises(OutOfRangeError, Diss_soliton, position=1.1)
        self.assertRaises(OutOfRangeError, Diss_soliton, width=1e3)
        self.assertRaises(OutOfRangeError, Diss_soliton, peak_power=1e9)
        self.assertRaises(OutOfRangeError, Diss_soliton, offset_nu=200.0)
        self.assertRaises(OutOfRangeError, Diss_soliton, C=1e3)
        self.assertRaises(OutOfRangeError, Diss_soliton, initial_phase=2.0 * pi)
        self.assertRaises(OutOfRangeError, Diss_soliton, channel=2)

    def test_wrong_type(self):
        """ Should fail if wrong type """
        self.assertRaises(NotIntegerError, Diss_soliton, channel=0.5)


class CheckInputAndOutput(unittest2.TestCase):
    """ Test input and output of Sech. """
    def test_output(self):
        """ Check Diss_soliton outputs its values correctly """
        dsoliton = Diss_soliton("diss_soliton", 0.2, 5.0, 1.4, 0.3, 0, 0.4, 0.0, 1)
        expected_string = [
            'position = 0.200000', 'width = 5.000000 ps',
            'fwhm = 8.813736 ps', 'peak_power = 1.400000 W',
            'offset_nu = 0.300000 THz', 'C = 0.400000',
            'initial_phase = 0.000000 rad', 'channel = 1']
        self.assertEqual(str(dsoliton), '\n'.join(expected_string))


class CheckFunctions(unittest2.TestCase):
    """ Test class methods. """
    def test_conversion(self):
        """ Should calculate a FWHM pulse width from the HWIeM value. """
        dsoliton = Diss_soliton(width=100.0)
        fwhm = dsoliton.calculate_fwhm()
        self.assertEqual(fwhm, 176.2747174039086)

    def test_bad_t(self):
        """ Should raise exception when temporal array has too few values """
        from numpy import arange
        t = arange(0.0, 4.0)
        self.assertRaises(OutOfRangeError, Diss_soliton().generate, t)

    @staticmethod
    def test_call():
        """ Check generated Diss_soliton function """
        from numpy import arange
        dsoliton = Diss_soliton("diss_soliton", 0.4, 5.0, 1.5, 0.3, 0, 1.5)
        t = arange(0.0, 100.0)
        A = dsoliton.generate(t)
        P = abs(A) ** 2
        assert_almost_equal(max(P), 1.5)

if __name__ == "__main__":
    unittest2.main()
