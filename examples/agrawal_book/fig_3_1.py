
"""
    Copyright (C) 2011, 2012  David Bolt

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

from pyofss import Domain, System, Gaussian, Fibre
from pyofss import temporal_power, chirp, multi_plot, labels

P_ts = []
chirps = []
zs = [0.0, 2.0, 4.0]

for z in zs:
    system = System(Domain(bit_width=200.0, samples_per_bit=2048))
    system.add(Gaussian(peak_power=1.0, width=1.0))
    system.add(Fibre(length=z, beta=[0.0, 0.0, -1.0, 0.0]))
    system.run()

    field = system.fields['fibre']
    P_ts.append(temporal_power(field))

    temp = [f if abs(f) >= 5e-12 else 0.0 for f in field]
    chirps.append(chirp(temp, system.domain.window_nu))

multi_plot(system.domain.t, P_ts, zs, labels["t"], labels["P_t"],
           [r"$z = {0:.0f} \, km$"], (-10.0, 10.0), filename="3-1")

multi_plot(system.domain.t, chirps, zs,
           labels["t"], labels["chirp"], [r"$z = {0:.0f} \, km$"],
           (-10.0, 10.0), (-6.0, 6.0), filename="3-1_chirp")
