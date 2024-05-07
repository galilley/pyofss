
"""
    Copyright (C) 2012  David Bolt, 2023 Vladislav Efremov

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

import numpy as np
import math

from scipy.interpolate import interpn

from pyofss import field
from pyofss.field import temporal_power
from pyofss.field import spectral_power

try:
    import pyopencl.array as pycl_array
except ImportError:
    print("OpenCL is not activated, Storage will work with NumPy arrays only")


def reduce_to_range(x, ys, first_value, last_value):
    """
    :param array_like x: Array of x values to search
    :param array_like ys: Array of y values corresponding to x array
    :param first_value: Initial value of required range
    :param last_value: Final value of required range
    :return: Reduced x and y arrays
    :rtype: array_like, array_like

    From a range given by first_value and last_value, attempt to reduce
    x array to required range while also reducing corresponding y array.
    """
    print("Attempting to reduce storage arrays to specified range...")
    if last_value > first_value:
        def find_nearest(array, value):
            """ Return the index and value closest to those provided. """
            index = (np.abs(array - value)).argmin()
            return index, array[index]

        first_index, x_first = find_nearest(x, first_value)
        last_index, x_last = find_nearest(x, last_value)

        print("Required range: [{0}, {1}]\nActual range: [{2}, {3}]".format(
            first_value, last_value, x_first, x_last))

        # The returned slice does NOT include the second index parameter. To
        # include the element corresponding to last_index, the second index
        # parameter should be last_index + 1:
        sliced_x = x[first_index:last_index + 1]

        # ys is a list of arrays. Does each array contain additional arrays:
        import collections
        if isinstance(ys[0][0], collections.Iterable):
            sliced_ys = [[y_c0[first_index:last_index + 1],
                          y_c1[first_index:last_index + 1]]
                         for (y_c0, y_c1) in ys]
        else:
            sliced_ys = [y[first_index:last_index + 1] for y in ys]

        return sliced_x, sliced_ys
    else:
        print("Cannot reduce storage arrays unless last_value > first_value")


class Storage(object):
    """
    Contains A arrays for multiple z values. Also contains t array and
    functions to modify the stored data.
    """
    def __init__(self, length, traces):
        self.length = length
        self.traces = traces
        self.trace_zs = []
        if self.traces < 1:
            self.trace_zs = 2*[length]
        elif self.traces == 1:
            self.trace_zs = [length]
        else:
            self.trace_zs = np.linspace(0.0, self.length, self.traces + 1)
        self.trace_n = 0
        self.t = []
        self.As = []
        self.z = []

        self.buff_As = []
        self.buff_z = []

        self.nu = []

        # List of tuples of the form (z, h); one tuple per successful step:
        self.step_sizes = []

        # Accumulate number of fft and ifft operations used for a stepper run:
        self.fft_total = 0

    def reset_array(self):
        self.trace_n = 0
        self.As = []
        self.z = []

    @staticmethod
    def reset_fft_counter():
        """ Resets the global variable located in the field module. """
        field.fft_counter = 0

    def store_current_fft_count(self):
        """ Store current value of the global variable in the field module. """
        self.fft_total = field.fft_counter

    def get_A(self, A):
        if isinstance(A, np.ndarray):
            return A
        elif isinstance(A, pycl_array.Array):
            return A.get()
        else:
            raise TypeError("unsupported type {} is stored".format(type(A)))

    def append(self, z, A):
        """
        :param double z: Distance along fibre
        :param array_like A: Field at distance z

        Append current fibre distance and field to stored array
        """
        if self.traces < 1:
            self.z.append(z)
            self.As.append(self.get_A(A))
        # Проверяем, совпадает ли текущая длина с точкой сохранения
        # если совпадает, сохраняем поле, переходим к следующей точке сохранения
        # и добавляем в буффер
        elif math.isclose(self.trace_zs[self.trace_n], z):
            self.trace_n += 1

            self.z.append(z)
            self.As.append(self.get_A(A))

            self.buff_z = [z]
            self.buff_As = [self.get_A(A)]
        # если не совпадает, проверяем, не прошли ли точку сохранения
        # если прошли, далаем интерполяцию на основе буффера
        # сохраняем последние значения из него и переходим к след. точке сохранения
        elif self.trace_zs[self.trace_n] < z:
            self.buff_z.append(z)
            self.buff_As.append(self.get_A(A))

            # проверяем, есть ли ещё точки сохранения, которые мы перешагнули
            zs = [zs for zs in self.trace_zs[self.trace_n:] if zs < z]
            self.trace_n = np.where(np.isclose(self.trace_zs, zs[-1]))[0][0]

            self.interpolate_As_for_z_values(zs)

            self.buff_z = [self.buff_z[-1]]
            self.buff_As = [self.buff_As[-1]]

            self.trace_n += 1
        # если не прошли точку сохранения, отправляем текущие значения в буффер
        # и идём дальше
        elif self.traces != 1:
            self.buff_z.append(z)
            self.buff_As.append(self.get_A(A))

    def get_plot_data(self, is_temporal=True, reduced_range=None,
                      normalised=False, channel=None):
        """
        :param bool is_temporal: Use temporal domain data (else spectral
                                 domain)
        :param Dvector reduced_range: Reduced x_range. Reduces y array to
                                      match.
        :param bool normalised: Normalise y array to first value.
        :param Uint channel: Channel number if using WDM simulation.
        :return: Data for x, y, and z axis
        :rtype: Tuple

        Generate data suitable for plotting. Includes temporal/spectral axis,
        temporal/spectral power array, and array of z values for the x,y data.
        """
        if is_temporal:
            x = self.t
            calculate_power = temporal_power
        else:
            x = self.nu
            calculate_power = spectral_power

        if channel is not None:
            temp = [calculate_power(A[channel]) for A in self.As]
        else:
            temp = [calculate_power(A) for A in self.As]

        if normalised:
            factor = max(temp[0])
            y = [t / factor for t in temp]
        else:
            y = temp

        if reduced_range is not None:
            x, y = reduce_to_range(x, y, reduced_range[0], reduced_range[1])

        z = np.array(self.z)

        return (x, y, z)

    @staticmethod
    def find_nearest(array, value):
        """
        :param array_like array: Array in which to locate value
        :param double value: Value to locate within array
        :return: Index and element of array corresponding to value
        :rtype: Uint, double
        """
        index = (np.abs(array - value)).argmin()
        return index, array[index]

    def interpolate_As_for_z_values(self, zs):
        """
        :param array_like zs: Array of z values for which A is required

        Split into separate arrays, interpolate each, then re-join.
        """
        import collections

        # Check if As[0] is itself a list of iterable elements, e.g.
        # As[0] = [ [A_0, B_0], [A_1, B_1], ... , [A_N-1, B_N-1] ]
        # rather than just a list of (non-iterable) elements, e.g.
        # As[0] = [ A_0, A_1, ... , A_N-1 ]
        if isinstance(self.buff_As[0][0], collections.Iterable):
            # Separate into channel_0 As and channel_1 As:
            As_c0, As_c1 = list(zip(*self.buff_As))

            As_c0 = self.interpolate_As(zs, As_c0)
            As_c1 = self.interpolate_As(zs, As_c1)

            # Interleave elements from both channels into a single array:
            self.As = list(zip(As_c0, As_c1))
        else:
            self.As.extend(self.interpolate_As(zs, self.buff_As))

        # Finished using original z; can now overwrite with new values (zs):
        self.z.extend(zs)

    def interpolate_As(self, zs, As):
        """
        :param array_like zs: z values to find interpolated A
        :param array_like As: Array of As to be interpolated
        :return: Interpolated As
        :rtype: array_like

        Interpolate array of A values, stored at non-uniform z-values, over a
        uniform array of new z-values (zs).
        """
        from scipy.interpolate import barycentric_interpolate, pchip_interpolate
        As = np.array(As)
        if As[0].dtype.name.startswith('complex'):
            As1_r = barycentric_interpolate(self.buff_z, np.real(As), zs)
            As1_i = barycentric_interpolate(self.buff_z, np.imag(As), zs)
            As = As1_r + 1j*As1_i
        else:
            As = barycentric_interpolate(self.buff_z, As, zs)
        return As


if __name__ == "__main__":
    # Compare simulations using Fibre and OpenclFibre modules.
    from pyofss import Domain, System, Gaussian, Fibre
    from pyofss import multi_plot, double_plot, labels

    import time
    import matplotlib.pyplot as plt

    domain = Domain(bit_width=200.0, total_bits=8, samples_per_bit=512 * 32)
    gaussian = Gaussian(peak_power=1.0, width=1.0)

    traces = np.arange(7, 201, 11)

    fibers = {f'{key}': Fibre(
                         length=20,
                         method='rk4ip',
                         total_steps=200,
                         traces=key,
                         beta=[0.0, 0.0, 0.0, 1.0],
                         gamma=1.5,
                         use_all='hollenbeck'
                         ) for key in traces}

    dur = []
    max_rel_err = []
    A_t = {}

    for numb in traces:
        sys = System(domain)
        sys.add(gaussian)
        sys.add(fibers[f'{numb}'])

        start = time.time()
        sys.run()
        stop = time.time()

        dur.append(stop - start)
        stor = fibers[f'{numb}'].stepper.storage
        A1 = stor.As[-int(numb/2)]
        A_t[f'{numb}'] = temporal_power(A1)
        z1 = stor.z[-int(numb/2)]
        print(z1)

        fiber0 = Fibre(length=z1, method='rk4ip', total_steps=300, traces=1,
                       beta=[0.0, 0.0, 0.0, 1.0], gamma=1.5, use_all='hollenbeck')
        sys0 = System(domain)
        sys0.add(gaussian)
        sys0.add(fiber0)
        sys0.run()
        A_t0 = temporal_power(sys0.field)

        DELTA_POWER = A_t0 - A_t[f'{numb}']

        MEAN_RELATIVE_ERROR = np.mean(np.abs(DELTA_POWER))
        MEAN_RELATIVE_ERROR /= np.max(A_t0)

        MAX_RELATIVE_ERROR = np.max(np.abs(DELTA_POWER))
        MAX_RELATIVE_ERROR /= np.max(A_t0)
        max_rel_err.append(MAX_RELATIVE_ERROR)
        print(f"Run time with {numb} traces: {dur[-1]}")
        print(f"Mean relative error: {MEAN_RELATIVE_ERROR}")
        print(f"Max relative error: {MAX_RELATIVE_ERROR}")

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(traces, dur, 'g-.', label='Duration')
    ax2 = fig.add_subplot(212)
    ax2.plot(traces, max_rel_err, "r-s", label='Max relative err')
    ax2.set_ylabel('Max relative err')
    ax2.set_xlabel('Traces')

    ax1.set_xlabel('Traces')
    ax1.set_ylabel('Duration, sec')
    plt.tight_layout()

    plt.savefig('trace_comp')
