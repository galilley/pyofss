
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
from scipy import exp, power, log
from pyofss.field import fft, ifft, ifftshift
import matplotlib.pyplot as plt
from pyofss import *


# Define exceptions
class FilterError(Exception):
    pass


class OutOfRangeError(FilterError):
    pass


class NotIntegerError(FilterError):
    pass


class Filter_doubl(object):
    """
    :param string name: Name of this module
    :param double width_nu1: Spectral bandwidth of first
                                filter (HWIeM). *Unit: THz*
    :param double width_nu2: Spectral bandwidth of second
                                filter (HWIeM). *Unit: THz*
    :param double relation: Ratio of filter amplitudes
    :param double offset_nu: Offset frequency relative to domain centre
                            frequency. *Unit: THz*
    :param Uint m: Order parameter. m > 1 describes a super-Gaussian filter
    :param Uint channel: Channel of the field array to modify if multi-channel.
    :param bool using_fwhm: Determines whether the width_nu parameter is a
                            full-width at half-maximum measure (FWHM), or
                            a half-width at 1/e-maximum measure (HWIeM).
    :param type: specify which radiation will be used next

    Generate a double super-Gaussian filter. A HWIeM bandwidth is used internally; a
    FWHM bandwidth will be converted on initialisation.
    """
    def __init__(self, name="filter_double", 
                 width_nu1=0.1, width_nu2=1.0,
                 relation=2. ,offset_nu=0.0,
                 m=1, channel=0, using_fwhm=False,
                 type_filt = "reflected"):

        if not (1e-6 < width_nu1 < 1e3):
            raise OutOfRangeError(
                "width_nu is out of range. Must be in (1e-6, 1e3)")

        if not (1e-6 < width_nu2 < 1e3):
            raise OutOfRangeError(
                "width_nu is out of range. Must be in (1e-6, 1e3)")

        if not (-200.0 < offset_nu < 200.0):
            raise OutOfRangeError(
                "offset_nu is out of range. Must be in (-200.0, 200.0)")

        if not (0 < m < 50):
            raise OutOfRangeError(
                "m is out of range. Must be in (0, 50)")

        if not (0 <= channel < 2):
            raise OutOfRangeError(
                "channel is out of range. Must be in [0, 2)")

        if not (relation > 0):
            raise OutOfRangeError("relation must be a positive")

        if int(m) != m:
            raise NotIntegerError("m must be an integer")

        if int(channel) != channel:
            raise NotIntegerError("channel must be an integer")

        self.name = name
        self.width_nu1 = width_nu1
        self.width_nu2 = width_nu2
        self.offset_nu = offset_nu
        self.m = m
        self.channel = channel
        self.fwhm_nu = None
        self.type = type_filt

        self.a1 = 1./(relation + 1.)
        self.a2 = relation/(relation + 1.)

        self.shape = None
        self.field = None

    def calculate_fwhm(self):
        """ Convert a HWIeM width to a FWHM width. """
        if self.fwhm_nu is not None:
            return self.fwhm_nu
        else:
            return self.width_nu * 2.0 * power(log(2.0), 1.0 / (2 * self.m))

    def __call__(self, domain, field):
        """
        :param object domain: A domain
        :param object field: Current field
        :return: Field after modification by Gaussian filter
        :rtype: Object
        """
        # Convert field to spectral domain:
        self.field = fft(field)

        delta_nu = domain.nu - domain.centre_nu - self.offset_nu
        factor1 = power(delta_nu / self.width_nu1, (2 * self.m))
        factor2 = power(delta_nu / self.width_nu2, (2 * self.m))
        # Frequency values are in order, inverse shift to put in fft order:
        self.shape = self.a1*exp(-0.5 * ifftshift(factor1)) + self.a2*exp(-0.5 * ifftshift(factor2))

        if domain.channels > 1:
            # Filter is applied only to one channel:
            self.field[self.channel] *= self.shape
        else:
            self.field *= self.shape

        # convert field back to temporal domain:
        if self.type == "reflected":
            return ifft(self.field)
        else:
            return field - ifft(self.field)

    def transfer_function(self, nu, centre_nu):
        """
        :param Dvector nu: Spectral domain array.
        :param double centre_nu: Centre frequency.
        :return: Array of values.
        :rtype: Dvector

        Generate an array representing the filter power transfer function.
        """
        if len(nu) < 8:
            raise OutOfRangeError(
                "Require spectral array with at least 8 values")

        delta_nu = nu - centre_nu - self.offset_nu
        factor1 = power(delta_nu / self.width_nu1, (2 * self.m))
        factor2 = power(delta_nu / self.width_nu2, (2 * self.m))
        self.shape = self.a1*exp(-0.5 * factor1) + self.a2*exp(-0.5 * factor2)

        return np.abs(self.shape) ** 2

    def __str__(self):
        """
        :return: Information string
        :rtype: string

        Output information on Filter.
        """
        output_string = [
            'amplitude1 = {0:f}', 'amplitude2 = {1:f}',
            'width_nu1 = {2:f} THz', 'width_nu2 = {3:f} THz',
            'offset_nu = {4:f} THz', 'm = {5:d}', 'channel = {6:d}',
            'type_filt = {7:s}']

        return "\n".join(output_string).format(
            self.a1, self.a2,
            self.width_nu1, self.width_nu2,
            self.offset_nu, self.m, self.channel,
            self.type)


if __name__ == "__main__":
    """ Plot the power transfer function of the filter """
    from pyofss import Domain, Filter_doubl, single_plot

    domain = Domain(centre_nu=193.0)
    gauss_filter = Filter_doubl(offset_nu=1.5)
    filter_tf = gauss_filter.transfer_function(domain.nu, domain.centre_nu)

    # Expect the filter transfer function to be centred at 194.5 THz:
    single_plot(domain.nu, filter_tf)
