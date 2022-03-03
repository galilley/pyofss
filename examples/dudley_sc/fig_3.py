
"""
    Copyright (C) 2022  Denis Kharenko

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

import sys
import numpy as np
from pyofss import Domain, System, Sech, Fibre
from pyofss.domain import lambda_to_nu, nu_to_lambda
from pyofss import map_plot, labels, single_plot, energy

def dB(num):
    return 10 * np.log10(num)

domain = Domain(bit_width=10.0, samples_per_bit=2**14,
                centre_nu=lambda_to_nu(835.0))

width = 0.050 # pulse duration 50 fs FWHM

beta = [0.0, 0.0, -11.830, 8.1038e-2, -9.5205e-5, 2.0737e-7, -5.3943e-10, 1.3486e-12,
        -2.5495e-15, 3.0524e-18, -1.7140e-21]

length = 150*1e-6 # length in km

P_0 = 10000 # peak power W
gamma = 110 # 1/(W km)

tau_shock = 0.56e-3 # 0.56 fs, corresponds to 1055.04 nm CWL

system = System(domain)
system.add(Sech(peak_power=P_0, width=width, using_fwhm=True))
system.add(Fibre(length=length, gamma=gamma, beta=beta, self_steepening=tau_shock, use_all='hollenbeck',
                 total_steps=1000, traces=300, method='ARK4IP', local_error=0.001))
system.run()


storage = system['fibre'].stepper.storage
(x, y, z_temp) = storage.get_plot_data(False, (lambda_to_nu(1350), lambda_to_nu(400)), True)
z_label = r"Fibre length, $z \, (cm)$"
z = z_temp * 1.0e5

map_plot(x, dB(y/np.max(y)), z, labels["nu"], labels["P_nu"], z_label,
         filename="fig_3_map_nu", y_range=(-40.0, 0))

single_plot(x, dB(y[-1]), labels["nu"], labels["P_nu"],
         filename="fig_3_nu", use_fill=False)

single_plot(nu_to_lambda(x), dB(y[-1]), labels["Lambda"], labels["P_lambda"],
         filename="fig_3_lambda", use_fill=False)

(x, y, z_temp) = storage.get_plot_data(reduced_range=(-1, 5))
z = z_temp * 1.0e5

map_plot(x, dB(y/np.max(y)), z, labels["t"], labels["P_t"], z_label,
        filename="fig_3_map_t", y_range=(-40.0, 0))

single_plot(x, y[-1], labels["t"], labels["P_t"],
         filename="fig_3_t", use_fill=False)
