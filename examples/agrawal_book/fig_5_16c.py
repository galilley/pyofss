
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

import sys
import numpy as np
from pyofss import Domain, System, Sech, Fibre
from pyofss import map_plot, waterfall_plot, animated_plot, labels

system = System(Domain(bit_width=100.0, samples_per_bit=2048))

absolute_separation = 3.5
offset = absolute_separation / system.domain.bit_width

system.add(Sech(peak_power=1.0, width=1.0, position=-offset))
system.add(Sech(peak_power=1.0, width=1.0, position=+offset,
                initial_phase=np.pi / 2.0))

system.add(Fibre(length=90.0, beta=[0.0, 0.0, -1.0, 0.0], gamma=1.0,
                 total_steps=200, traces=100, method='ARK4IP'))
system.run()

storage = system['fibre'].stepper.storage
(x, y, z) = storage.get_plot_data(reduced_range=(-10.0, 10.0))

map_plot(x, y, z, labels["t"], labels["P_t"], labels["z"],
         filename="5-16c_map")

waterfall_plot(x, y, z, labels["t"], labels["z"], labels["P_t"],
               filename="5-16c_waterfall", y_range=(0.0, 1.2))

if (len(sys.argv) > 1) and (sys.argv[1] == 'animate'):
    animated_plot(x, y, z, labels["t"], labels["P_t"], r"$z = {0:7.3f} \, km$",
                  (x[0], x[-1]), (0.0, 1.2), fps=20, frame_prefix="c_",
                  filename="5-16c_animation.avi")
