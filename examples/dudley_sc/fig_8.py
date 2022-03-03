
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

domain = Domain(bit_width=12.0, samples_per_bit=2**14,
                centre_nu=lambda_to_nu(835.0))

P_0 = 3480.      # peak power W
width = 0.010   # pulse duration 10 fs
gamma = 110     # (W km)^-1

beta_full = [0.0, 0.0, -11.830, 
           8.1038e-2,  -9.5205e-5,   2.0737e-7,  
          -5.3943e-10,  1.3486e-12, -2.5495e-15, 
           3.0524e-18, -1.7140e-21]


L_d = (width**2)/abs(beta_full[2])
L_nl = 1/(gamma*P_0)

z_sol = np.pi/2*L_d
print(" L_nl={:.2f} cm\n L_d={:.2f} cm\n z_sol={:.2f}cm\n N={:.2f}\n P={:.3f} kW".format(L_nl*1e5, L_d*1e5, z_sol*1e5, np.abs((L_d/L_nl))**0.5, P_0*1e-3))

length = 50*1e-5 # length in km

system = System(domain)
system.add(Sech(peak_power=P_0, width=width, using_fwhm=True))
system.add(Fibre(length=length, gamma=gamma, beta=beta_full, self_steepening=True, use_all='hollenbeck',
                 total_steps=100, traces=100, method='ARK4IP', local_error=0.005))
system.run()

storage = system['fibre'].stepper.storage
(x, y, z_temp) = storage.get_plot_data(False, (lambda_to_nu(1300), lambda_to_nu(500)), True)
z_label = r"Fibre length, $z \, (m)$"
z = z_temp * 1.0e3

map_plot(x, dB(y/np.max(y)), z, labels["nu"], labels["P_nu"], z_label,
         filename="fig_8_map_nu", y_range=(-40.0, 0))

single_plot(nu_to_lambda(x), y[-1], labels["Lambda"], labels["P_lambda"],
         filename="fig_8_lambda", use_fill=False)

(x, y, z_temp) = storage.get_plot_data(reduced_range=(-1, 4))
z = z_temp * 1.0e3

map_plot(x, dB(y/np.max(y)), z, labels["t"], labels["P_t"], z_label,
        filename="fig_8_map_t", y_range=(-40.0, 0))

single_plot(x, y[-1], labels["t"], labels["P_lambda"],
         filename="fig_8_t", use_fill=False)
