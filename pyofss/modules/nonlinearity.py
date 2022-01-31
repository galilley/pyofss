
"""
    Copyright (C) 2012  David Bolt, 2021-2022 Denis Kharenko

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
from numpy import pi
from scipy.constants import constants
import scipy.integrate as integrate

from pyofss.field import fft, ifft, fftshift
from pyofss.domain import Domain

class NonlinearityError(Exception):
    pass

class UnknownRamanResponseError(NonlinearityError):
    pass

def calculate_gamma(nonlinear_index, effective_area,
                    centre_omega=2.0 * pi * 193.1):
    """
    :param double nonlinear_index: :math:`n_2`. *Unit:* :math:`m^2 / W`
    :param double effective_area: :math:`A_{eff}`. *Unit:* :math:`\mu m^2`
    :param double centre_omega: Angular frequency of carrier. *Unit: rad / ps*
    :return: Nonlinear parameter. :math:`\gamma`. *Unit:* :math:`rad / (W km)`
    :rtype: double

    Calculate nonlinear parameter from the nonlinear index and effective area.
    """

    gamma = 1.0e24 * nonlinear_index * centre_omega / \
        (effective_area * Domain.vacuum_light_speed)

    return gamma


def calculate_raman_term(domain, tau_1=12.2e-3, tau_2=32.0e-3):
    """
    :param object domain: A domain
    :param double tau_1: First adjustable parameter. *Unit: ps*
    :param double tau_2: Second adjustable parameter. *Unit: ps*
    :return: Raman response function
    :rtype: double

    Calculate raman response function from tau_1 and tau_2 [1].

    [1] K. J. Blow and D. Wood, IEEE JQE 25, 2665 (1989) doi: 10.1109/3.40655
    """
    t = domain.t - domain.t.min() #shift starting point to zero
    h_R = (tau_2 ** 2 + tau_1 ** 2) / (tau_1 * tau_2 ** 2)
    h_R *= np.exp(-t / tau_2) * np.sin(t / tau_1)

    return h_R

def calculate_raman_term_silica(domain):
    """
    :param object domain: A domain
    :return: Raman response function
    :rtype: double

    Calculate raman response function according by Multiple-vibrational-mode model [1].
    The function is normalized such that its integral is unity [2].

    [1] D. Hollenbeck and C. D. Cantrell, JOSA B 19, 2886 (2002) doi: 10.1364/JOSAB.19.002886
    [2] R. H. Stolen et al, JOSA B 6, 1159 (1989) doi: 10.1364/JOSAB.6.001159
    """
    CP = [56.25, 100.00, 231.25, 362.50, 463.00, 497.00, 611.50, 691.67, 793.67, 
            835.50, 930.00, 1080.00, 1215.00]
    Ai = [1.00, 11.40, 36.67, 67.67, 74.00, 4.50, 6.80, 4.60, 4.20, 4.50, 2.70, 3.10, 3.00]
    Gfwhm = [52.10, 110.42, 175.00, 162.50, 135.33, 24.50, 41.50, 155.00, 59.50,
            64.30, 150.00, 91.00, 160.00]
    Lfwhm = [17.37, 38.81, 58.33, 54.17, 45.11, 8.17, 13.83, 51.67, 19.83, 21.43, 50.00, 30.33, 53.33]

    t = domain.t - domain.t.min() #shift starting point to zero
    h_R = np.zeros_like(t)
    for i in range(13):
        h_R += Ai[i] * \
                np.exp(-np.pi * 1e-10 * constants.c * Lfwhm[i] * t) * \
                np.exp(-(np.pi * 1e-10 * constants.c * Gfwhm[i]) ** 2 * (t ** 2) / 4.) * \
                np.sin(2 * np.pi * 1e-10 * constants.c * CP[i] * t)

    norm = integrate.simps(h_R, domain.t)
    return h_R/norm


class Nonlinearity(object):
    """
    Nonlinearity is used by fibre to generate a nonlinear factor.
    """
    def __init__(self, gamma=None, sim_type=None, self_steepening=False,
                 raman_scattering=False, rs_factor=3e-3, use_all=False,
                 tau_1=12.2e-3, tau_2=32.0e-3, f_R=0.18):

        self.gamma = gamma
        self.self_steepening = self_steepening
        self.raman_scattering = raman_scattering
        self.rs_factor = rs_factor
        self.use_all = use_all

        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.f_R = f_R

        self.generate_nonlinearity = getattr(
            self, "%s_nonlinearity" % sim_type, self.default_nonlinearity)

        if use_all:
            print("Using general expression for nonlinearity")
            self.non = getattr(self, "%s_f_all" % sim_type, self.default_f_all)
            self.exp_non = getattr(self, "%s_exp_f_all" % sim_type,
                                   self.default_exp_f_all)
        else:
            if self.self_steepening and self.raman_scattering:
                print("Using self_steepening and raman_scattering")
                self.non = getattr(self, "%s_f_with_ss_and_rs" % sim_type,
                                   self.default_f_with_ss_and_rs)
                self.exp_non = getattr(
                    self, "%s_exp_f_with_ss_and_rs" % sim_type,
                    self.default_exp_f_with_ss_and_rs)
            elif self.self_steepening:
                print("Using self_steepening")
                self.non = getattr(self, "%s_f_with_ss" % sim_type,
                                   self.default_f_with_ss)
                self.exp_non = getattr(self, "%s_exp_f_with_ss" % sim_type,
                                       self.default_exp_f_with_ss)
            elif self.raman_scattering:
                print("Using raman_scattering")
                self.non = getattr(self, "%s_f_with_rs" % sim_type,
                                   self.default_f_with_rs)
                self.exp_non = getattr(self, "%s_exp_f_with_rs" % sim_type,
                                       self.default_exp_f_with_rs)
            else:
                self.non = getattr(self, "%s_f" % sim_type,
                                   self.default_f)
                self.exp_non = getattr(self, "%s_exp_f" % sim_type,
                                       self.default_exp_f)

        self.ss_factor = None
        self.h_R = None
        self.omega = None
        self.centre_omega = None
        self.factor = None

    def __call__(self, domain):
        self.centre_omega = domain.centre_omega
        self.omega = fftshift(domain.omega - domain.centre_omega)

        if self.self_steepening is False:
            self.ss_factor = 0.0
        elif self.self_steepening is True:
            self.ss_factor = 1.0 / self.centre_omega
        else:
            self.ss_factor = float(self.self_steepening)

        if self.use_all is False:
            self.h_R = 0.0
        elif self.use_all is True or self.use_all.lower() == "blowwood": # use Blow Wood model by default
            # Require h_R in spectral domain, so take FFT of returned value:
            self.h_R = fft(calculate_raman_term(
                domain, self.tau_1, self.tau_2))
        elif self.use_all.lower() == "hollenbeck": # use Hollenbeck and Cantrell multiple-vibrational-mode model
            self.h_R = fft(calculate_raman_term_silica(domain))
        else:
            raise UnknownRamanResponseError(
                    "Use False, True or a sting from the list [\"blowwood\", \"hollenbeck\"]")

        # note: domain.window_t is added according to the case of Periodic convolution
        # https://en.wikipedia.org/wiki/Convolution_theorem
        self.h_R *= domain.window_t

        self.generate_nonlinearity()

    def default_nonlinearity(self):
        """ Set the common factor for default. """
        if self.gamma is None:
            self.factor = 0.0
        else:
            self.factor = 1j * self.gamma

    def wdm_nonlinearity(self):
        """ Set the common factor for WDM case. """
        if self.gamma is None:
            self.factor = (0.0, 0.0)
        else:
            self.factor = (1j * self.gamma[0], 1j * self.gamma[1])

    def default_f_all(self, A, z):
        """ Set all nonlinear terms. """
        term_spm = np.abs(A) ** 2
        convolution = ifft(self.h_R * fft(term_spm))
        p = fft(A * (1.0 - self.f_R) * term_spm + self.f_R * A * convolution)
        
        return ifft(self.factor * (1.0 + self.omega * self.ss_factor) * p) # 1j * -1j = 1

    def default_exp_f_all(self, A, h, B):
        """ Set all terms within an exponential factor. """
        term_spm = np.abs(A) ** 2
        convolution = ifft(self.h_R * fft(term_spm))
        #p = fft(A * (1.0 - self.f_R) * term_spm + self.f_R * A * convolution)
        term_all = (1.0 - self.f_R) * term_spm + self.f_R * convolution
        
        term_ss_all = np.where(np.abs(B) > 1e-15, self.ss_factor / B, 0.0) * ifft(self.omega * fft(term_all * A))

        #return np.exp(h * ifft(self.factor *
        #              (1.0 + self.omega * self.ss_factor) * p)) * B
        # TODO gamma can only be a constant!!!
        return np.exp(h * self.factor * (term_all + term_ss_all)) * B

    def default_f_with_ss(self, A, z):
        """ Use self-steepening only. """
        term_spm = np.abs(A) ** 2 * A
        term_ss = self.ss_factor * ifft(self.omega * fft(term_spm))

        return self.factor * (term_spm + term_ss)

    def default_exp_f_with_ss(self, A, h, B):
        """ Use self-steepening in exponential term. """
        term_spm = np.abs(A) ** 2
        term_ss = (self.ss_factor / B) * ifft(self.omega * fft(term_spm * A))
        return np.exp(h * self.factor * (term_spm + term_ss)) * B

    def default_f_with_rs(self, A, z):
        """ Use Raman-scattering term. """
        term_spm = np.abs(A) ** 2
        term_rs = self.rs_factor * ifft(1j * self.omega * fft(term_spm))

        return self.factor * (term_spm + term_rs) * A

    def default_exp_f_with_rs(self, A, h, B):
        """ Use Raman-scattering in exponential term. """
        term_spm = np.abs(A) ** 2
        term_rs = self.rs_factor * ifft(1j * self.omega * fft(term_spm))

        return np.exp(h * self.factor * (term_spm + term_rs)) * B

    def default_f_with_ss_and_rs(self, A, z):
        """ Use self-steepening and Raman-scattering terms. """
        term_spm = np.abs(A) ** 2
        term_ss = self.ss_factor * ifft(self.omega * fft(term_spm * A))
        term_rs = self.rs_factor * ifft(1j * self.omega * fft(term_spm))

        return self.factor * (term_spm + term_rs) * A + term_ss

    def default_exp_f_with_ss_and_rs(self, A, h, B):
        " Use self-steepening and Raman-scattering within exponential term. """
        term_spm = np.abs(A) ** 2
        term_ss = (self.ss_factor / B) * ifft(self.omega * fft(term_spm * A))
        term_rs = self.rs_factor * ifft(1j * self.omega * fft(term_spm))

        return np.exp(h * self.factor * (term_spm + term_ss + term_rs)) * B

    def default_f(self, A, z):
        return self.factor * np.abs(A) ** 2 * A

    def default_exp_f(self, A, h, B):
        return np.exp(h * self.factor * np.abs(A) ** 2) * B

    def wdm_f_with_ss(self, As, z):
        self.wdm_f(As, z)

    def wdm_exp_f_with_ss(self, As, h, Bs):
        self.wdm_exp_f(As, h, Bs)

    def wdm_f_with_rs(self, As, z):
        self.wdm_f(As, z)

    def wdm_exp_f_with_rs(self, As, h, Bs):
        self.wdm_exp_f(As, h, Bs)

    def wdm_f_with_ss_and_rs(self, As, z):
        self.wdm_f(As, z)

    def wdm_exp_f_with_ss_and_rs(self, As, h, Bs):
        self.wdm_exp_f(As, h, Bs)

    def wdm_f(self, As, z):
        return np.asarray(
            [self.factor[0] *
             (np.abs(As[0]) ** 2 + 2.0 * np.abs(As[1]) ** 2) * As[0],
             self.factor[1] *
             (np.abs(As[1]) ** 2 + 2.0 * np.abs(As[0]) ** 2) * As[1]])

    def wdm_exp_f(self, As, h, Bs):
        return np.asarray(
            [np.exp(h * self.factor[0] *
             (np.abs(As[0]) ** 2 + 2.0 * np.abs(As[1]) ** 2)) * Bs[0],
             np.exp(h * self.factor[1] *
             (np.abs(As[1]) ** 2 + 2.0 * np.abs(As[0]) ** 2)) * Bs[1]])
