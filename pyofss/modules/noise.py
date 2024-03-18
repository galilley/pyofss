
"""
    Copyright (C) 2022 Denis Kharenko

    This file is part of pyofss-2.

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

from numpy import pi, exp, random
from scipy.constants import constants
from pyofss.field import ifft, fftshift

# Functions are deprecated and will be removed in SciPy 2.0.0
try:
    from numpy.lib.scimath import sqrt, log
except ImportError:  #try import old syntax for compatibility reason
    from scipy import sqrt, log


# Define exceptions
class NoiseError(Exception):
    pass


class OutOfRangeError(NoiseError):
    pass


class Noise(object):
    """
    :param string name: Name of this module
    :param string model: Determines the model of noise
                         opm - one photon per mode (standard quantum
                            noise [arXiv:1902.01273v1])

    Generates a noise defined by selected model. 
    """

    def __init__(self, name="noise", model="opm"):
        self.name = name
        self.model = model
        self.fwhm = None

        self.field = None

    def __call__(self, domain, field):
        """
        :param object domain: A domain
        :param object field: Current field
        :return: Field with added noise
        :rtype: Object
        """
        self.field = field

        phase = 2 * pi * random.ranf(domain.total_samples)
        magnitude = sqrt(constants.h * domain.nu / domain.dnu) # TODO normalization?

        if domain.channels > 1:
            self.field[self.channel] += ifft(fftshift(magnitude * exp(1j * phase)))
        else:
            self.field += ifft(fftshift(magnitude * exp(1j * phase)))

        return self.field

    def __str__(self):
        """
        :return: Information string
        :rtype: string

        Output information on noise.
        """

        output_string = [
            'model = {0:s}']

        return "\n".join(output_string).format(
            self.model)

if __name__ == "__main__":
    """ Plot a default noise in temporal and spectral domain """
    from pyofss import Domain, System, Noise
    from pyofss import temporal_power, spectral_power
    from pyofss import double_plot, labels

    sys = System(Domain(bit_width=500.0))
    sys.add(Noise())
    sys.run()

    double_plot(sys.domain.t, temporal_power(sys.field),
                sys.domain.nu, spectral_power(sys.field, True),
                labels["t"], labels["P_t"], labels["nu"], labels["P_nu"])
