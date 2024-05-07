# -*- coding: utf-8 -*-

"""
    Copyright (C) 2022  Vladislav Efremov, Denis Kharenko

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

import os
import datetime
import json
import time
import collections

from pyofss import *

from scipro.field import Field
from scipro.reader import pyofss as pfr
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import pylab as plt
#plt.switch_backend('Qt5Agg')
plt.switch_backend('agg')

steps_per_km_smf = 100000

dtype = 'double'
pathtosave = "data/"
rtcount = 0

#################### Main settings of RDS generator ########################

lag = 56.8                              # Time delay. Unit: ps
p_power = 100.0                         # Peak power of pump pulse. Unit: W
wav = 1550.0                            # Central wavelength of pump pulse. Unit: nm
dur = 18.8                              # Pump pulse duration. Unit: ps
chirp = 100                             # Chirp
coupler_coef = 0.8                      # feedback coefficient (Coupler losses)

length_PM1550 = 0.5*0.001               # Length of PM1550 fibre. Unit: km
length_PMR = 28.788*0.001               # Length of PM Raman fibre. Unit: km

############################################################################

series = 'example'                      # name of the series
par_roundnum = 100                      # number of roundtrip

def build_file_path(suff=None, pref=None):
    """
    [pref_]Series[str]_PeakPower[W]_Delay[ps][_suff]
    """
    s = "s={}_lag={:.1f}_pp={}".format(series, lag, p_power)
    if suff is not None:
        s = s + "_" + suff;
    if pref is not None:
        s = pref + "_" + s;
    return s


def initsystem(domain, field = None, params = {}):
    global lag, p_power, wav, dur, chirp, coupler_coef, par_roundnum

    fibre_ctx = OpenclFibre()

    # Add PM1550 fibre
    fibre_PM1550_0m5_1 = OpenclFibre(
        name="PM1550_0m5_1",
        length=length_PM1550,
        total_steps=int(steps_per_km_smf*length_PM1550),
        beta=[0.0, 0.0, -22.07, 0.0178],
        gamma=1.6,
        use_all='hollenbeck',
        self_steepening=True,
        dorf=dtype,
        ctx=fibre_ctx.ctx,
        )

    # Add PM Raman fibre
    fibre_PMR = OpenclFibre(
        name="PMR",
        length=length_PMR,
        total_steps=int(steps_per_km_smf*length_PMR),
        beta=[0.0, 0.0, 26.0, -0.01],
        gamma=6.9,
        use_all='hollenbeck',
        self_steepening=True,
        dorf=dtype,
        ctx=fibre_ctx.ctx,
        )

    # Add PM1550 fibre
    fibre_PM1550_0m5_2 = OpenclFibre(
        name="PM1550_0m5_2",
        length=length_PM1550,
        total_steps=int(steps_per_km_smf * length_PM1550),
        beta=[0.0, 0.0, -22.07, 0.0178],
        gamma=1.6,
        use_all='hollenbeck',
        self_steepening=True,
        dorf=dtype,
        ctx=fibre_ctx.ctx,
        )

    # Add two pieces PM1550 fibre
    fibre_PM1550_1m = OpenclFibre(
        name="PM1550_1m",
        length=length_PM1550 * 2,
        total_steps=int(steps_per_km_smf * length_PM1550 * 2),
        beta=[0.0, 0.0, -22.07, 0.0178],
        gamma=1.6,
        use_all='hollenbeck',
        self_steepening=True,
        dorf=dtype,
        ctx=fibre_ctx.ctx,
        )

    сoupler = Splitter('сoupler', loss=coupler_coef)

    delay = Delay("delay", lag)

    pump = Diss_soliton(
        name = 'pump',
        width= dur,
        offset_nu = align_with_nu_grid(lambda_to_nu(wav) - domain.centre_nu, domain),
        peak_power=p_power,
        C = chirp,
        using_fwhm=True
        )

    wdm = StaticPumpWDM("wdm", "lowpass", lambda_to_nu(1600),
        pump(domain, np.zeros(domain.t.size, dtype=np.complex64)))

    sys = System(domain, field)
    sys.add(wdm)
    sys.add(fibre_PM1550_0m5_1)
    sys.add(fibre_PMR)
    sys.add(fibre_PM1550_0m5_2)
    sys.add(сoupler)
    sys.add(fibre_PM1550_1m)
    sys.add(delay)
    return sys

def simulate():
    global series

    Ep = Pp = Wp = 0.
    Ec = Pc = Wc = 1.

    Ecol = collections.deque(maxlen=10)

    # data to save
    issavefulldata = True
    issavenarrowspectrum = True

    par_roundstart = 0

    #OpenclFibre.print_device_info()

    # Create domain of system
    domain = Domain(
        total_bits=8*4,
        samples_per_bit=512 * 64,
        bit_width=100.0,
        centre_nu=lambda_to_nu(1550.0)
        )

    # Check availability of the data and/or create a folder for saving
    if not os.path.exists(pathtosave):
        os.mkdir(pathtosave)

    filepath = os.path.join(pathtosave, build_file_path())

    p_power_n = 1000

    if os.path.exists(filepath):
        try:
            A = field_load(filepath + "/field_out")
        except:
            print("Warning: can not find full field, try load narrow spectrum")
            try:
                As = field_load(filepath + "/field_spectrum_narrow")
                fn = Field( domain.nu, As, d='freq', cf=domain.centre_nu ).ifft()
                A = pfr.field_convert_back(fn)[0]
            except:
                print("Warning: can not find narrow field, start will be with noise")
                A = (p_power_n ** 0.5) * 1e-5 * (np.pi * np.random.ranf(domain.total_samples) - 0.5) * np.exp(1j * 2 * np.pi * np.random.ranf(domain.total_samples))
    else:
        os.mkdir(filepath)
        A = (p_power_n ** 0.5) * 1e-5 * (np.pi * np.random.ranf(domain.total_samples) - 0.5) * np.exp(1j * 2 * np.pi * np.random.ranf(domain.total_samples))

    ######################### System initialization ############################

    # creating the system and sending the initial field and parameters
    fwmsys = initsystem(domain, A)

    now = datetime.datetime.now()

    file_E=os.path.join(filepath, 'energy.txt')
    with open(file_E,'a') as f:
        f.write('%s'%(now.strftime("%d-%m-%Y %H:%M:%S.%f\n")))

    # Organize a dictionary for saving parameters and results
    dres = {}

    dres['domain'] = {
            'total_bits': domain.total_bits,
            'samples_per_bit': domain.samples_per_bit,
            'bit_width': domain.bit_width,
            'centre_lambda': round(nu_to_lambda(domain.centre_nu))
            }

    dres['lag'] = lag
    dres['p_power'] = p_power

    wdm2 = StaticPumpWDM("wdm2", "lowpass", lambda_to_nu(1600))
    wdmsys = System(domain)
    wdmsys.add(wdm2)
    wdmsys.run()

    estcnt = 0
    field2save = None

    i = par_roundstart

    while i < par_roundstart + par_roundnum:
        t0 = time.time()

        # start the calculation
        fwmsys.run()

        fout = wdmsys['wdm2'](domain, fwmsys.fields['сoupler'].copy())
        Ec = energy(fout, fwmsys.domain.t)
        Ecol.append(Ec)

        with open(file_E,'a') as f:
            if i==0:
                f.write('%(T)s\t%(RUN)d\t%(Energy)G\n' %{"T":datetime.datetime.now().strftime("%H:%M:%S.%f"), "RUN":i,"Energy": Ec})
            else:
                f.write('%(T)s\t%(RUN)d\t%(Energy)G\t%(Rep)G\n' % {"T":datetime.datetime.now().strftime("%H:%M:%S.%f"), "RUN" : i, "Energy" : Ec, "Rep" : ((Ec-Ep)/Ec)})

        # Check if the operation mode is set
        if np.abs((Ec-Ep)/Ec) < 1e-4:
            estcnt += 1
        else:
            estcnt = 0

        i += 1

        # exit if the operation mode is set
        if estcnt > Ecol.maxlen:
            print("Energy settled, calculation complete")
            dres['settled'] = True
            if Ec < 1e-7:
                field2save = None
            else:
                field2save = fwmsys.field
            break

        # exit if the mode fades out
        if Ec < 1e-7 and i > 200:
            print("Energy becames to small")
            dres['settled'] = True
            field2save = None
            break

        # exit if a calculation error has occurred
        if not np.isfinite(Ec):
            print("Energy is infinite, abort")
            # save the field from the last roundtrip
            field2save = fwmsys.field
            break

        field2save = fwmsys.field
        Ep = Ec

        print("END of round {} in {:.2f} secs".format(i, time.time() - t0))

    dres['par_roundnum'] = i

    if field2save is not None:
        fout = wdmsys['wdm2'](domain, fwmsys.fields['сoupler'].copy())
        Ec = energy(fout, fwmsys.domain.t)
        dres['Eout'] = Ec
        dres['Estd'] = np.array(Ecol).std()
        f = pfr.field_convert(fout, domain.window_t, domain.centre_nu)
        osc = f.abs2()
        dres['T'] = osc.bandwidth()
        dres['Ppeak'] = osc.y.max()
        if issavefulldata:
            field_save(field2save, filename=os.path.join(filepath, 'field_out'))
        if issavenarrowspectrum:
            sfp = pfr.field_convert( field2save, domain.window_t, domain.centre_nu ).fft()
            sfpn = sfp.split_filled( lambda_to_nu(1800), lambda_to_nu(1400) )[1]
            field_save(sfpn.y.astype(np.complex64), filename=os.path.join(filepath, 'field_spectrum_narrow'))

    file_r = os.path.join(filepath, 'result.json')
    with open(file_r,'w') as f:
        json.dump(dres, f, indent=4)

    return fwmsys


if __name__ == "__main__":
    simulate()