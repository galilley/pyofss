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

import matplotlib as mpl

pathtosave = "data/"
iscont = False
steps_per_km = 6250  # started from 100000
dtype = 'double'
rtcount = 0


#################### Main settings of FOPO ########################

width_pl = 80.        # Half width pump puplse at fwhm,. Unit: ps

p_power = 1100.       # Peak power of pump pulse. Unit: W
lag = 85.0            # Time delay. Unit: ps

#p_power = 1500.0
#lag = 135.0

#p_power = 1400.0
#lag = 50.0

length_pcf = 0.001/3  # Length of photonic crystal fiber. Unit: km
length_780 = 0.050    # Length of passive fiber. Unit: km

###################################################################

# continuation of the previous calculation
iscont = False
# number of roundtrip
par_roundnum = 500
# name of the series
series = 'examp'


def build_file_name(baseName, suff=None, pref=None):
    """
    [pref_]Series[num]_PeakPower[W]_Delay[ps]_baseName[_suff]
    """
    s = "s{}_pp={}_d={:+.3f}_{}".format(series, int(p_power), lag, baseName)
    if suff is not None:
        s = s + "_" + suff;
    if pref is not None:
        s = pref + "_" + s;
    return s


def build_file_path(suff=None, pref=None):
    """
    [pref_]Series[str]_PeakPower[W]_Delay[ps][_suff]
    """
    s = "s={}_pw={:.1f}_pp={}_d={:+.3f}".format(series, width_pl, int(p_power), lag)
    if suff is not None:
        s = s + "_" + suff;
    if pref is not None:
        s = pref + "_" + s;
    return s


def initsystem(domain, field = None):
    global lag, p_power, width_pl

    wavelength = 1032

    pump = Gaussian(name = 'pump',
            width=width_pl,
            offset_nu = align_with_nu_grid(lambda_to_nu(wavelength) - domain.centre_nu, domain),
            peak_power=p_power,
            using_fwhm=True
            )

    delay = Delay("delay", lag)

    # typical losses for silica fiber
    loss_log = loss_infrared_db(domain.Lambda) + loss_rayleigh_db(domain.Lambda)
    loss_lin = fftshift(convert_alpha_to_linear( loss_log ))
    #loss_lin = 0

    fibre_ctx = OpenclFibre(
            )

    fibreforward = OpenclFibre(
            name="fibre_f",
            length=length_pcf,
            total_steps=30,
            alpha=loss_lin,
            beta=[0.0, 0.0, 0.0, 6.756*1e-2, -1.002*1e-4, 3.671*1e-7],  # Slow axis, ZDW = 1051.85, Zlobina JOSA B 29 1959 (2012)
            #beta=[0.0, 0.0, 0.0, 6.753e-2, -1.001e-4, 2.753e-7],  # Fast axis, ZDW = 1052.95
            gamma=10.0,
            centre_omega=lambda_to_omega(1052.0),
            dorf=dtype,
            ctx=fibre_ctx.ctx,
            )

    fibrebackward = OpenclFibre(
            name="fibre_b",
            length=length_pcf,
            total_steps=15,
            alpha=loss_lin,
            beta=[0.0, 0.0, 0.0, 6.756*1e-2, -1.002*1e-4, 3.671*1e-7],  # Slow axis, LMA5-PM PCF, ZDW = 1051.85, Zlobina JOSA B 29 1959 (2012)
            #beta=[0.0, 0.0, 0.0, 6.753e-2, -1.001e-4, 2.753e-7],  # Fast axis, ZDW = 1052.95
            gamma=10.0,
            centre_omega=lambda_to_omega(1052.0),
            dorf=dtype,
            ctx=fibre_ctx.ctx,
            )

    # Add PM - 780 fibre
    fibre_780 = OpenclFibre(
            name="fibre_780",
            length=length_780,
            total_steps=int(steps_per_km*length_780),
            alpha=loss_lin,
            beta=[0.0, 0.0, 53., 0.132], # J Liu et al Optical Engineering, 2012 DOI:10.1117/1.OE.51.8.085001
            gamma=4.0,
            centre_omega=lambda_to_omega(780.0),
            dorf=dtype,
            ctx=fibre_ctx.ctx,
            )

    wdmf = StaticPumpWDM("wdm_f", "highpass", lambda_to_nu(900), 
            pump(domain, np.zeros(domain.t.size, dtype=np.complex64)))
    wdmb = StaticPumpWDM("wdm_b", "highpass", lambda_to_nu(900))

    split1 = Splitter('split1', loss=0.96)
    #split2 = Splitter('split2', loss=0.3)
    split2 = Splitter('split2', loss=0.0)
    split3 = Splitter('split3', loss=0.1) 

    sys = System(domain, field)
    sys.add(wdmf)
    sys.add(fibreforward)
    sys.add(split1)
    sys.add(fibrebackward)
    sys.add(wdmb)
    #sys.add(split2)
    sys.add(fibre_780)
    sys.add(split3)
    sys.add(fibre_780)
    #sys.add(split2)
    sys.add(delay)
    return sys

def roundtrips(sys, N=1):
    global rtcount
    for i in range(N):
        sys.run()
        rtcount += 1

def simulate(pd):
    global series, iscont, length_pcf, length_780

    par_roundstart = 0

    # if the previous calculations continue
    if iscont:
        with open(pd + "/result.json",'r') as fd:
            dr = json.load(fd)
        par_roundstart = dr['Rnum']
        length_pcf = dr['Lpcf']
        length_780 = dr['L780']
           
    #OpenclFibre.print_device_info()

    domain = Domain(
        total_bits=32,
        samples_per_bit=512 * 64 * 2,
        bit_width=110.0,
        centre_nu=lambda_to_nu(760.0)
        )
    
    # continuation of the previous count or start from noise
    if iscont:
        try:
            A = field_load(pd + "/field_out")
        except:
            print("Error: can not find full field")
            return
    else:
        A = (p_power ** 0.5) * 1e-5 * (np.pi * np.random.ranf(domain.total_samples) - 0.5) * np.exp(1j * 2 * np.pi * np.random.ranf(domain.total_samples))

    # creating the system and sending the initial field and parameters
    fwmsys = initsystem(domain, A)
    
    now = datetime.datetime.now()
    if not os.path.exists(pathtosave):
        os.mkdir(pathtosave)

    filepath = os.path.join(pathtosave, build_file_path(suff='pcf'))
    if not iscont:
        if os.path.exists(os.path.join(filepath, 'result.json')):
            print("Error: path \'{}\' already exists! Exit...".format(filepath))
            return
        
    file_E=os.path.join(filepath, 'energy.txt')
    with open(file_E,'a') as f:
        f.write('%s'%(now.strftime("%d-%m-%Y %H:%M:%S.%f\n")))

    Ep = 0.
    Ec = 1.
    Ecol = collections.deque(maxlen=70)

    dres = {}
    if iscont and par_roundnum < Ecol.maxlen:
        dres['settled'] = dr['settled']
    else:
        dres['settled'] = False

    dres['domain'] = {
            'total_bits': domain.total_bits,
            'samples_per_bit': domain.samples_per_bit,
            'bit_width': domain.bit_width,
            'centre_lambda': round(nu_to_lambda(domain.centre_nu))
            }
    dres['Lpcf'] = length_pcf
    dres['L780'] = length_780

    dres['Tpump'] = width_pl
    dres['Ppump'] = p_power
    dres['delay'] = lag
    
    estcnt = 0
    field2save = None
    i = par_roundstart
    while i < par_roundstart + par_roundnum:
        t0 = time.time()

        # start the calculation
        fwmsys.run()

        Ec = energy(fwmsys.fields['delay'], fwmsys.domain.t)
        Ecol.append(Ec)

        with open(file_E,'a') as f:
            if i==0:
                f.write('%(T)s\t%(RUN)d\t%(Energy)G\n' %{"T":datetime.datetime.now().strftime("%H:%M:%S.%f"), "RUN":i,"Energy": Ec})
            else:
                f.write('%(T)s\t%(RUN)d\t%(Energy)G\t%(Rep)G\n' % {"T":datetime.datetime.now().strftime("%H:%M:%S.%f"), "RUN" : i, "Energy" : Ec, "Rep" : ((Ec-Ep)/Ec)})
        
        # check if the operation mode is set
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
        if Ec < 1e-7 and i > 300:
            print("Energy becames to small")
            dres['settled'] = True
            field2save = None
            break

        # exit if a calculation error has occurred
        if not np.isfinite(Ec):
            print("Energy is infinite, abort")
            break;

        field2save = fwmsys.field
        Ep = Ec
        
        print("END of round {} in {:.2f} secs".format(i, time.time() - t0))

    dres['Rnum'] = i

    if field2save is not None:
        fout = fwmsys['wdm_b'](domain, fwmsys.fields['fibre_f'].copy())
        Ec = energy(fout, fwmsys.domain.t)
        dres['Eout'] = Ec
        dres['Estd'] = np.array(Ecol).std()
        field_save(field2save, filename=os.path.join(filepath, 'field_out'))

    file_r = os.path.join(filepath, 'result.json')
    with open(file_r,'w') as f:
        json.dump(dres, f, indent=4)

if __name__ == "__main__":
    if not os.path.exists(pathtosave):
        os.mkdir(pathtosave)
    filepath = os.path.join(pathtosave, build_file_path(suff='pcf'))
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    simulate(filepath)
