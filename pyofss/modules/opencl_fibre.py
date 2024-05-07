
"""
    Copyright (C) 2013 David Bolt,
    2020-2021, 2023 Vladislav Efremov, Denis Kharenko

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
import pyopencl as cl
import pyopencl.array as cl_array

from scipy import linalg

import sys as sys0
version_py = sys0.version_info[0]
if version_py == 3:
    from reikna import cluda
    from reikna.fft import FFT
else:
    from pyfft.cl import Plan  
from string import Template

from .linearity import Linearity
from .nonlinearity import Nonlinearity
from .storage import Storage

OPENCL_OPERATIONS = Template("""
    #ifdef cl_khr_fp64 // Khronos extension
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        #define PYOPENCL_DEFINE_CDOUBLE
    #elif defined(cl_amd_fp64) // AMD extension
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #define PYOPENCL_DEFINE_CDOUBLE
    #else
        #warning "Double precision floating point is not supported"
    #endif

    #include <pyopencl-complex.h>

    __kernel void cl_cache(__global c${dorf}_t* factor,
                           const ${dorf} stepsize) {
        int gid = get_global_id(0);

        factor[gid] = c${dorf}_exp(c${dorf}_mulr(factor[gid], stepsize));
    }
    
    __kernel void cl_factor(__global c${dorf}_t* buf_factor,
                           __global c${dorf}_t* factor,
                           const ${dorf} stepsize) {
        int gid = get_global_id(0);

        buf_factor[gid] = c${dorf}_exp(c${dorf}_mulr(factor[gid], stepsize));
    }

    __kernel void cl_linear_cached(__global c${dorf}_t* field,
                                   __global c${dorf}_t* factor) {
        int gid = get_global_id(0);

        field[gid] = c${dorf}_mul(field[gid], factor[gid]);
    }

    __kernel void cl_linear(__global c${dorf}_t* field,
                            __global c${dorf}_t* factor,
                            const ${dorf} stepsize) {
        int gid = get_global_id(0);

        // IMPORTANT: Cannot just multiply two complex numbers!
        // Must use the appropriate function (e.g. cfloat_mul).
        field[gid] = c${dorf}_mul(field[gid],
                                  c${dorf}_exp(c${dorf}_mulr(factor[gid], stepsize)));
    }

    c${dorf}_t cl_square_abs(const c${dorf}_t element) {
        return c${dorf}_mul(element, c${dorf}_conj(element));
    }

    __kernel void cl_square_abs2(__global c${dorf}_t* field) {
        int gid = get_global_id(0);
        field[gid] = c${dorf}_new(c${dorf}_abs_squared(field[gid]), (${dorf})0.0f);
    }

    __kernel void cl_norm(__global c${dorf}_t* field,
                          __global c${dorf}_t* err) {
        int i = get_global_id(0);
        err[i] = c${dorf}_mul(field[i], c${dorf}_conj(field[i]));
    }

    __kernel void cl_nonlinear(__global c${dorf}_t* field,
                               const ${dorf} gamma,
                               const ${dorf} stepsize) {
        int gid = get_global_id(0);

        const c${dorf}_t im_gamma = c${dorf}_new((${dorf})0.0f, stepsize * gamma);

        field[gid] = c${dorf}_mul(
            field[gid], c${dorf}_mul(im_gamma, cl_square_abs(field[gid])));
    }
    
    __kernel void cl_nonlinear_exp(__global c${dorf}_t* fieldA,
                               const __global c${dorf}_t* fieldB,
                               const ${dorf} gamma,
                               const ${dorf} stepsize) {
        int gid = get_global_id(0);

        const c${dorf}_t im_gamma = c${dorf}_new((${dorf})0.0f, stepsize * gamma);

        fieldA[gid] = c${dorf}_mul(
            fieldB[gid], c${dorf}_exp(
                c${dorf}_mul(im_gamma, cl_square_abs(fieldA[gid]))));
    }

    __kernel void cl_sum(__global c${dorf}_t* first_field,
                         const ${dorf} first_factor,
                         __global c${dorf}_t* second_field,
                         const ${dorf} second_factor) {
        int gid = get_global_id(0);

        first_field[gid] = c${dorf}_mulr(first_field[gid], first_factor);
        first_field[gid] = c${dorf}_add(first_field[gid], c${dorf}_mulr(second_field[gid], second_factor));
    }

    __kernel void cl_mul(__global c${dorf}_t* field,
                         __global c${dorf}_t* factor) {
        int gid = get_global_id(0);

        field[gid] = c${dorf}_mul(field[gid], factor[gid]);
    }

    __kernel void cl_nonlinear_with_all_st1(__global c${dorf}_t* field,
                                          __global c${dorf}_t* field_mod,
                                          __global c${dorf}_t* conv,
                                          const ${dorf} fR,
                                          const ${dorf} fR_inv) {
        int gid = get_global_id(0);

        conv[gid] = c${dorf}_new(c${dorf}_real(conv[gid]), (${dorf})0.0f);

        field[gid] = c${dorf}_add(
            c${dorf}_mul(c${dorf}_mulr(field[gid], fR_inv), field_mod[gid]),
                c${dorf}_mul(c${dorf}_mulr(field[gid], fR), conv[gid]));

    }

    __kernel void cl_nonlinear_with_all_st2_with_ss(__global c${dorf}_t* field,
                                __global c${dorf}_t* factor,
                                const ${dorf} gamma) {
        int gid = get_global_id(0);

        const c${dorf}_t im_gamma = c${dorf}_new((${dorf})0.0f, gamma);

        field[gid] = c${dorf}_mul(
            im_gamma, c${dorf}_mul(
                field[gid], factor[gid]));
    }

    __kernel void cl_step_mul(__global c${dorf}_t* field,
                                const ${dorf} stepsize) {
        int gid = get_global_id(0);

        field[gid] = c${dorf}_mulr(
            field[gid], stepsize);
    }

    __kernel void cl_nonlinear_with_all_st2_without_ss(__global c${dorf}_t* field,
                                const ${dorf} gamma) {
        int gid = get_global_id(0);

        const c${dorf}_t im_gamma = c${dorf}_new((${dorf})0.0f, gamma);

        field[gid] = c${dorf}_mul(
            im_gamma, field[gid]);
    }

    __kernel void cl_split(__global c${dorf}_t* field,
                        __global c${dorf}_t* subfield1, __global c${dorf}_t* subfield2,
                        const int n)
    {
        int gid = get_global_id(0);
        if (gid < n)
        {
            subfield1[gid] = field[gid];
        }
        else
        {
            subfield2[gid-n] = field[gid];
        }
    }

    __kernel void cl_couple(__global c${dorf}_t* field,
                        __global c${dorf}_t* subfield1, __global c${dorf}_t* subfield2,
                        const int n)
    {
        int gid = get_global_id(0);
        if (gid < n)
        {
            field[gid] = subfield1[gid];
        }
        else
        {
            field[gid] = subfield2[gid-n];
        }
    }

""")

errors = {"cl_ss_symmetric": 3,
          "cl_ss_sym_rk4": 5, "cl_rk4ip": 5}


# Define exceptions
class StepperError(Exception):
    pass


class SmallStepSizeError(StepperError):
    pass


class SuitableStepSizeError(StepperError):
    pass


class MaximumStepsAllocatedError(StepperError):
    pass


class OpenclFibre(object):
    """
    This optical module is similar to Fibre, but uses PyOpenCl (Python
    bindings around OpenCL) to generate parallelised code.
    method:
        * cl_rk4ip
        * cl_ss_symmetric
        * cl_ss_sym_rk4
    """
    def __init__(self, name="ocl_fibre", length=1.0, alpha=None,
                 beta=None, gamma=0.0, traces=1,
                 local_error=1.0e-6, method="cl_rk4ip", total_steps=100,
                 self_steepening=False,
                 use_all=False, centre_omega=None,
                 tau_1=12.2e-3, tau_2=32.0e-3, f_R=0.18,
                 channels=1, dorf='double', ctx=None, fast_math=False):

        self.name = name
        
        self.domain = None

        self.channels = channels

        self.use_all = use_all

        self.gamma = gamma

        self.nn_factor = None
        self.buf_nn_factor = None
        self.h_R = None
        self.buf_h_R = None
        self.omega = None
        self.ss_factor = None

        self.queue = None
        self.np_float = None
        self.np_complex = None
        self.prg = None
        self.compiler_options = None
        self.fast_math = fast_math
        self.ctx = ctx
        self.cl_initialise(dorf)

        self.plan = None

        self.buf_field = None
        self.subfield1 = None
        self.subfield2 = None
        self.buf_temp = None
        self.buf_interaction = None
        self.buf_factor = None

        self.buf_mod = None
        self.buf_conv = None

        self.f_R = f_R
        self.f_R_inv = 1.0 - f_R

        self.shape = None
        self.plan = None

        self.cached_factor = False
        # Force usage of cached version of function:
        self.cl_linear = self.cl_linear_cached

        self.length = length
        self.total_steps = total_steps

        self.stepsize = self.length / self.total_steps
        self.zs = np.linspace(0.0, self.length, self.total_steps + 1)

        # Use a list of tuples ( z, A(z) ) for dense output if required:
        self.traces = traces
        self.storage = Storage(self.length, self.traces)

        # Check if adaptive stepsize is required:
        if method.upper().startswith('A'):
            self.adaptive = True
            self.local_error = local_error
            self.step_sizes = []
            self.z_adapt = []
            # Store constants for adaptive method:
            self.total_attempts = 100
            self.steps_max = 50000
            self.step_size_min = 1e-37  # some small value
            self.safety = 0.9
            self.max_factor = 10.0
            self.min_factor = 0.2
            # Store local error of method:
            self.eta = errors[method[1:].lower()]
            self.method = getattr(self, method[1:].lower())
            self.cl_linear = self.cl_linear_ad
        else:
            self.adaptive = False
            self.method = getattr(self, method.lower())

        if self.use_all:
            if self_steepening is False:
                self.cl_n = getattr(self, 'cl_n_with_all')
            else:
                self.cl_n = getattr(self, 'cl_n_with_all_and_ss')
        elif self_steepening:
            raise NotImplementedError("Self-steepening without general nonlinearity is not implemented")
        else:
            self.cl_n = getattr(self, 'cl_n_default')
        
        self.linearity = Linearity(alpha, beta, sim_type="default",
                                   use_cache=True, centre_omega=centre_omega, phase_lim=True)
        self.nonlinearity = Nonlinearity(gamma, self_steepening=self_steepening,
                                         use_all=use_all, tau_1=tau_1, tau_2=tau_2, f_R=f_R)
        self.factor = None

    def __call__(self, domain, field):
        # Setup plan for calculating fast Fourier transforms:
        if self.plan is None:
            if version_py == 3:
                self.plan = FFT(domain.t.astype(self.np_complex)).compile(self.thr,
                        fast_math=self.fast_math, compiler_options=self.compiler_options)
                if self.channels == 2:
                    self.plan.execute = self.reikna_fft_execute_ch2
                else:
                    self.plan.execute = self.reikna_fft_execute
            else:
                self.plan = Plan(domain.total_samples, queue=self.queue, dtype=self.np_complex, fast_math=self.fast_math)

        if self.domain != domain:
            self.domain = domain
            if self.use_all:
                self.nonlinearity(self.domain)
                self.omega = self.nonlinearity.omega
                self.h_R = self.nonlinearity.h_R
                self.ss_factor = self.nonlinearity.ss_factor
                if self.ss_factor != 0.0:
                    self.nn_factor = 1.0 + self.omega * self.ss_factor
                else:
                    self.nn_factor = None
                if self.channels == 2:
                    self.nn_factor = np.array([self.nn_factor, self.nn_factor])
                    self.h_R = np.array([self.h_R, self.h_R])

        if self.factor is None:
            factor = self.linearity(domain)
            if self.channels == 2:
                self.factor = np.array([factor, factor])
            else:
                self.factor = factor

        self.storage.t = domain.t
        self.storage.nu = domain.nu
        self.storage.reset_fft_counter()
        self.storage.reset_array()
        self.storage.append(0.0, field)

        if self.adaptive:
            return self.adaptive_stepper(field)
        else:
            return self.standard_stepper(field)

    def cl_initialise(self, dorf="float"):
        """ Initialise opencl related parameters. """
        float_conversions = {"float": np.float32, "double": np.float64}
        complex_conversions = {"float": np.complex64, "double": np.complex128}

        self.np_float = float_conversions[dorf]
        self.np_complex = complex_conversions[dorf]

        for platform in cl.get_platforms():
            if platform.name == "NVIDIA CUDA" and self.fast_math is True:
                print("Using compiler optimisations suitable for Nvidia GPUs")
                self.compiler_options = "-cl-mad-enable -cl-fast-relaxed-math"
            else:
                self.compiler_options = ""

        if self.ctx is None:
            self.ctx = cl.create_some_context()

        self.queue = cl.CommandQueue(self.ctx)
        if version_py == 3:
            api = cluda.ocl_api()
            self.thr = api.Thread(self.queue)

        substitutions = {"dorf": dorf}
        code = OPENCL_OPERATIONS.substitute(substitutions)
        self.prg = cl.Program(self.ctx, code).build(options=self.compiler_options)

    def standard_stepper(self, field):
        """ Take a fixed number of steps, each of equal length """
        
        self.send_arrays_to_device(field, self.factor)

        for z in self.zs[:-1]:
            self.method(self.buf_field, self.buf_temp,
                          self.buf_interaction, self.buf_factor, self.stepsize)
            self.storage.append(z + self.stepsize, self.buf_field)
        return self.buf_field.get()

    def adaptive_stepper(self, field):
        #### Статья, где хорошо объясняется адпативный шаг, а также
        #### исследуется его модифицированая версия:
        #### DOI:10.1109/JLT.2009.2021538

        # Constants used for approximation of solution using local
        # extrapolation:
        f_eta = np.power(2, self.eta - 1.0)
        f_alpha = f_eta / (f_eta - 1.0)
        f_beta = 1.0 / (f_eta - 1.0)
        fh_eta = np.power(2, 1/self.eta)

        z = 0.0
        self.z_adapt.append(z)
        h = self.stepsize

        self.send_arrays_to_device(field, self.factor)

        # Limit the number of steps in case of slowly converging runs:
        for s in range(1, self.steps_max):
            # If step-size takes z our of range [0.0, length], then correct it:
            if (z + h) > self.length:
                h = self.length - z

            # Take an adaptive step:
            for ta in range(0, self.total_attempts):
                h_half = 0.5 * h
                z_half = z + h_half

                self.cl_copy(self.buf_fine, self.buf_field)

                self.cl_copy(self.buf_coarse, self.buf_field)

                # Calculate fine solution using two steps of size h_half:
                self.cached_factor = False
                self.method(self.buf_fine, self.buf_temp,
                            self.buf_interaction, self.buf_factor, h_half)
                self.method(self.buf_fine, self.buf_temp,
                            self.buf_interaction, self.buf_factor, h_half)
                # Calculate coarse solution using one step of size h:
                self.cached_factor = False
                self.method(self.buf_coarse, self.buf_temp,
                            self.buf_interaction, self.buf_factor, h)

                delta = self.relative_local_error(self.buf_fine, self.buf_coarse)

                # Store current stepsize:
                h_temp = h

                # Adjust stepsize for next step:
                if delta > 0.0:
                    error_ratio = (self.local_error / delta)
                    factr = \
                        self.safety * np.power(error_ratio, 1.0 / self.eta)
                    h = h_temp * min(self.max_factor,
                                     max(self.min_factor, factr))
                else:
                    # Error approximately zero, so use largest stepsize
                    # increase:
                    h = h_temp * self.max_factor

                if delta < 2.0 * self.local_error:
                    # Successful step, so increment z h_temp (which is the
                    # stepsize that was used for this step):
                    z += h_temp

                    self.buf_field = self.buf_fine.mul_add(f_alpha, self.buf_coarse, -f_beta)

                    self.storage.append(z, self.buf_field)
                    self.step_sizes.append(h_temp)
                    self.z_adapt.append(z)

                    break  # Successful attempt at step, move on to next step.

                # Otherwise error was too large, continue with next attempt,
                # but check the minimal step size first
                else:
                    # h = h_temp/2
                    if h < self.step_size_min:
                        raise SmallStepSizeError("Step size is extremely small")

            else:
                raise SuitableStepSizeError("Failed to set suitable step-size")

            # If the desired z has been reached, then finish:
            if z >= self.length:
                return self.buf_field.get()

        raise MaximumStepsAllocatedError("Failed to complete with maximum steps allocated")


    def relative_local_error(self, A_fine, A_coarse):
        """ Calculate an estimate of the relative local error """
        self.prg.cl_norm(self.queue, self.shape, None,
                                A_fine.data, self.err.data)

        norm_fine = np.sqrt(np.real(cl_array.sum(self.err, queue=self.queue).get()))
        A_rel = A_fine.mul_add(1.0, A_coarse, -1.0, queue=self.queue)
        self.prg.cl_norm(self.queue, self.shape, None,
                         A_rel.data, self.err.data)
        norm_rel = np.sqrt(np.real(cl_array.sum(self.err, queue=self.queue).get()))

        # Avoid possible divide by zero:
        if norm_fine != 0.0:
            return norm_rel / norm_fine
        else:
            return norm_rel

    @staticmethod
    def print_device_info():
        """ Output information on each OpenCL platform and device. """
        for platform in cl.get_platforms():
            print("=" * 60)
            print("Platform information:")
            print("Name: ", platform.name)
            print("Profile: ", platform.profile)
            print("Vender: ", platform.vendor)
            print("Version: ", platform.version)

            for device in platform.get_devices():
                print("-" * 60)
                print("Device information:")
                print("Name: ", device.name)
                print("Type: ", cl.device_type.to_string(device.type))
                print("Memory: ", device.global_mem_size // (1024 ** 2), "MB")
                print("Max clock speed: ", device.max_clock_frequency, "MHz")
                print("Compute units: ", device.max_compute_units)

            print("=" * 60)

    def send_arrays_to_device(self, field, factor):
        """ Move numpy arrays onto compute device. """
        if self.channels == 2:
            self.shape = (field.shape[0]*field.shape[1], )
        else:
            self.shape = field.shape

        self.buf_field = cl_array.to_device(
            self.queue, field.astype(self.np_complex))

        if self.channels == 2:
            self.subfield1 = cl_array.empty(self.queue, (field.shape[1], ), self.np_complex)
            self.subfield2 = cl_array.empty(self.queue, (field.shape[1], ), self.np_complex)

        if self.buf_temp is None:
            self.buf_temp = cl_array.empty_like(self.buf_field)
        if self.buf_interaction is None:
            self.buf_interaction = cl_array.empty_like(self.buf_field)

        if self.use_all:
            if self.buf_h_R is None:
                self.buf_h_R = cl_array.to_device(
                                        self.queue, self.h_R.astype(self.np_complex))
            if self.buf_nn_factor is None and self.nn_factor is not None:
                self.buf_nn_factor = cl_array.to_device(
                                        self.queue, self.nn_factor.astype(self.np_complex))
            if self.buf_mod is None:
                self.buf_mod = cl_array.empty_like(self.buf_field)
            if self.buf_conv is None:
                self.buf_conv = cl_array.empty_like(self.buf_field)

        if self.cached_factor is False:
            self.buf_factor = cl_array.to_device(
                self.queue, factor.astype(self.np_complex))
        
        if self.adaptive:
            self.buf_only_factor = cl_array.to_device(
                    self.queue, factor.astype(self.np_complex))
            self.err = cl_array.empty_like(self.buf_field)
            self.buf_fine = cl_array.empty_like(self.buf_field)
            self.buf_coarse = cl_array.empty_like(self.buf_field)

    def cl_copy(self, dst_buffer, src_buffer):
        """ Copy contents of one buffer into another. """
        cl.enqueue_copy(self.queue, dst_buffer.data, src_buffer.data).wait()

    def cl_linear(self, field_buffer, stepsize, factor_buffer):
        """ Linear part of step. """
        self.plan.execute(field_buffer.data, inverse=True)
        self.prg.cl_linear(self.queue, self.shape, None, field_buffer.data,
                           factor_buffer.data, self.np_float(stepsize))
        self.plan.execute(field_buffer.data)
        
    def cl_linear_ad(self, field_buffer, stepsize, factor_buffer):
        """ Linear part of step. """
        if (self.cached_factor is False):
            self.prg.cl_factor(self.queue, self.shape, None, self.buf_factor.data,
                               self.buf_only_factor.data, self.np_float(stepsize))
            self.cached_factor = True
        
        self.plan.execute(field_buffer.data, inverse=True)
        self.prg.cl_linear_cached(self.queue, self.shape, None, field_buffer.data,
                                  self.buf_factor.data)
        self.plan.execute(field_buffer.data)

    def cl_linear_cached(self, field_buffer, stepsize, factor_buffer):
        """ Linear part of step (cached version). """
        if (self.cached_factor is False):
            self.linearity.cache(stepsize)
            if self.channels == 2:
                self.buf_factor = cl_array.to_device(
                    self.queue, np.array([self.linearity.cached_factor.astype(self.np_complex), self.linearity.cached_factor.astype(self.np_complex)]))
            else:
                self.buf_factor = cl_array.to_device(
                    self.queue, self.linearity.cached_factor.astype(self.np_complex))
            self.cached_factor = True
        self.plan.execute(field_buffer.data, inverse=True)
        self.prg.cl_linear_cached(self.queue, self.shape, None,
                                  field_buffer.data, self.buf_factor.data)
        self.plan.execute(field_buffer.data)

    def cl_n_default(self, field_buffer, stepsize):
        """ Nonlinear part of step. """
        self.prg.cl_nonlinear(self.queue, self.shape, None, field_buffer.data,
                              self.np_float(self.gamma), self.np_float(stepsize))
    
    def cl_nonlinear(self, fieldA_buffer, stepsize, fieldB_buffer):
        """ Nonlinear part of step, exponential term"""
        self.prg.cl_nonlinear_exp(self.queue, self.shape, None, fieldA_buffer.data, fieldB_buffer.data,
                                  self.np_float(self.gamma), self.np_float(stepsize))

    def cl_n_with_all_and_ss(self, field_buffer, stepsize):
        """ Nonlinear part of step with self_steepening and raman """
        # |A|^2
        self.cl_copy(self.buf_mod, field_buffer)
        self.prg.cl_square_abs2(self.queue, self.shape, None,
                                self.buf_mod.data)

        # conv = ifft(h_R*(fft(|A|^2)))
        self.cl_copy(self.buf_conv, self.buf_mod)
        self.plan.execute(self.buf_conv.data, inverse=True)
        self.prg.cl_mul(self.queue, self.shape, None,
                        self.buf_conv.data, self.buf_h_R.data)
        self.plan.execute(self.buf_conv.data)

        # p = fft( A*( (1-f_R)*|A|^2 + f_R*conv ) )
        self.prg.cl_nonlinear_with_all_st1(self.queue, self.shape, None,
                                           field_buffer.data, self.buf_mod.data, self.buf_conv.data,
                                           self.np_float(self.f_R), self.np_float(self.f_R_inv))
        self.plan.execute(field_buffer.data, inverse=True)

        # A_out = ifft( factor*(1 + omega*ss_factor)*p)
        self.prg.cl_nonlinear_with_all_st2_with_ss(self.queue, self.shape, None, field_buffer.data,
                              self.buf_nn_factor.data, self.np_float(self.gamma))
        self.plan.execute(field_buffer.data)

        self.prg.cl_step_mul(self.queue, self.shape, None,
                             field_buffer.data, self.np_float(stepsize))

    def cl_n_with_all(self, field_buffer, stepsize):
        """ Nonlinear part of step with self_steepening and raman """
        # |A|^2
        self.cl_copy(self.buf_mod, field_buffer)
        self.prg.cl_square_abs2(self.queue, self.shape, None,
                                self.buf_mod.data)

        # conv = ifft(h_R*(fft(|A|^2)))
        self.cl_copy(self.buf_conv, self.buf_mod)
        self.plan.execute(self.buf_conv.data, inverse=True)
        #print( max( temporal_power( self.buf_conv.get()[0] ) ), max( temporal_power( self.buf_conv.get()[1] ) ) )
        self.prg.cl_mul(self.queue, self.shape, None,
                        self.buf_conv.data, self.buf_h_R.data)
        self.plan.execute(self.buf_conv.data)

        # p = A*( (1-f_R)*|A|^2 + f_R*conv )
        self.prg.cl_nonlinear_with_all_st1(self.queue, self.shape, None,
                                           field_buffer.data, self.buf_mod.data, self.buf_conv.data,
                                           self.np_float(self.f_R), self.np_float(self.f_R_inv))

        # A_out = factor*p
        self.prg.cl_nonlinear_with_all_st2_without_ss(self.queue, self.shape, None, field_buffer.data,
                                                  self.np_float(self.gamma))

        self.prg.cl_step_mul(self.queue, self.shape, None,
                             field_buffer.data, self.np_float(stepsize))


    def cl_sum(self, first_buffer, first_factor, second_buffer, second_factor):
        """ Calculate weighted summation. """
        self.prg.cl_sum(self.queue, self.shape, None,
                        first_buffer.data, self.np_float(first_factor),
                        second_buffer.data, self.np_float(second_factor))

    def reikna_fft_execute(self, d, inverse=False):
        self.plan(d, d, inverse=inverse)

    def reikna_fft_execute_ch2(self, d, inverse=False):
        self.prg.cl_split(self.queue, self.shape, None,
                       d, self.subfield1.data, self.subfield2.data,
                       np.int32(self.shape[0]/2))
        self.plan(self.subfield1.data, self.subfield1.data, inverse=inverse)
        self.plan(self.subfield2.data, self.subfield2.data, inverse=inverse)
        self.prg.cl_couple(self.queue, self.shape, None,
                       d, self.subfield1.data, self.subfield2.data,
                       np.int32(self.shape[0]/2))
    
    def cl_ss_symmetric(self, field, field_temp, field_interaction, factor, stepsize):
        """ Symmetric split-step method using OpenCL"""
        half_step = 0.5 * stepsize
        self.cl_copy(field_temp, field)

        self.cl_linear(field_temp, half_step, factor)
        self.cl_nonlinear(field, stepsize, field_temp)
        self.cl_linear(field, half_step, factor)
    
    def cl_ss_sym_rk4(self, field, field_temp, field_linear, factor, stepsize):
        """ 
        Runge-Kutta fourth-order method using OpenCL.

            A_L = f.linear(A, hh)
            k0 = h * f.n(A_L, z)
            k1 = h * f.n(A_L + 0.5 * k0, z + hh)
            k2 = h * f.n(A_L + 0.5 * k1, z + hh)
            k3 = h * f.n(A_L + k2, z + h)
            A_N =  A_L + (k0 + 2.0 * (k1 + k2) + k3) / 6.0
            return f.linear(A_N, hh)
        """
        inv_six = 1.0 / 6.0
        inv_three = 1.0 / 3.0
        half_step = 0.5 * stepsize

        self.cl_linear(field, half_step, factor)  # A_L

        self.cl_copy(field_temp, field)
        self.cl_copy(field_linear, field)
        self.cl_n(field_temp, stepsize)  # k0
        self.cl_sum(field, 1, field_temp, inv_six)  # free k0

        self.cl_sum(field_temp, 0.5, field_linear, 1)
        self.cl_n(field_temp, stepsize)  # k1
        self.cl_sum(field, 1, field_temp, inv_three)  # free k1

        self.cl_sum(field_temp, 0.5, field_linear, 1)
        self.cl_n(field_temp, stepsize)  # k2
        self.cl_sum(field, 1, field_temp, inv_three)  # free k2

        self.cl_sum(field_temp, 1, field_linear, 1)
        self.cl_n(field_temp, stepsize)  # k3
        self.cl_sum(field, 1, field_temp, inv_six)  # free k3

        self.cl_linear(field, half_step, factor)

    def cl_rk4ip(self, field, field_temp, field_interaction, factor, stepsize):
        """ Runge-Kutta in the interaction picture method using OpenCL. """
        '''
        hh = 0.5 * h
        A_I = f.linear(A, hh)
        k0 = f.linear(h * f.n(A, z), hh)
        k1 = h * f.n(A_I + 0.5 * k0, z + hh)
        k2 = h * f.n(A_I + 0.5 * k1, z + hh)
        k3 = h * f.n(f.linear(A_I + k2, hh), z + h)
        return (k3 / 6.0) + f.linear(A_I + (k0 + 2.0 * (k1 + k2)) / 6.0, hh)
        '''

        inv_six = 1.0 / 6.0
        inv_three = 1.0 / 3.0
        half_step = 0.5 * stepsize

        self.cl_copy(field_temp, field)
        self.cl_linear(field, half_step, factor)

        self.cl_copy(field_interaction, field)  # A_I
        self.cl_n(field_temp, stepsize)
        self.cl_linear(field_temp, half_step, factor)  # k0

        self.cl_sum(field, 1.0, field_temp, inv_six)  # free k0
        self.cl_sum(field_temp, 0.5, field_interaction, 1.0)
        self.cl_n(field_temp, stepsize)  # k1

        self.cl_sum(field, 1.0, field_temp, inv_three)  # free k1
        self.cl_sum(field_temp, 0.5, field_interaction, 1.0)
        self.cl_n(field_temp, stepsize)  # k2

        self.cl_sum(field, 1.0, field_temp, inv_three)  # free k2
        self.cl_sum(field_temp, 1.0, field_interaction, 1.0)
        self.cl_linear(field_temp, half_step, factor)
        self.cl_n(field_temp, stepsize)  # k3

        self.cl_linear(field, half_step, factor)

        self.cl_sum(field, 1.0, field_temp, inv_six)


if __name__ == "__main__":
    # Compare simulations using Fibre and OpenclFibre modules.
    from pyofss import Domain, System, Gaussian, Sech, Fibre
    from pyofss import temporal_power, multi_plot, labels, lambda_to_nu

    import time
    
    print("*** Test of the default nonlinearity ***")
    # -------------------------------------------------------------------------
    TS = 4096*16
    GAMMA = 20.0
    BETA = [0.0, 0.0, 0.0, 22.0]
    STEPS = 800
    LENGTH = 0.1

    DOMAIN = Domain(bit_width=50.0, samples_per_bit=TS, centre_nu=lambda_to_nu(1050.0))

    SYS = System(DOMAIN)
    SYS.add(Gaussian("gaussian", peak_power=1.0, width=1.0))
    SYS.add(Fibre("fibre", beta=BETA, gamma=GAMMA,
                  length=LENGTH, total_steps=STEPS, method="RK4IP"))

    start = time.time()
    SYS.run()
    stop = time.time()
    NO_OCL_DURATION = (stop - start)
    NO_OCL_OUT = SYS.fields["fibre"]

    sys = System(DOMAIN)
    sys.add(Gaussian("gaussian", peak_power=1.0, width=1.0))
    sys.add(OpenclFibre("ocl_fibre", beta=BETA, gamma=GAMMA,
                        length=LENGTH, total_steps=STEPS))

    start = time.time()
    sys.run()
    stop = time.time()
    OCL_DURATION = (stop - start)
    OCL_OUT = sys.fields["ocl_fibre"]

    NO_OCL_POWER = temporal_power(NO_OCL_OUT)
    OCL_POWER = temporal_power(OCL_OUT)
    DELTA_POWER = NO_OCL_POWER - OCL_POWER

    MEAN_RELATIVE_ERROR = np.mean(np.abs(DELTA_POWER))
    MEAN_RELATIVE_ERROR /= np.max(temporal_power(NO_OCL_OUT))

    MAX_RELATIVE_ERROR = np.max(np.abs(DELTA_POWER))
    MAX_RELATIVE_ERROR /= np.max(temporal_power(NO_OCL_OUT))

    print("Run time without OpenCL: %e" % NO_OCL_DURATION)
    print("Run time with OpenCL: %e" % OCL_DURATION)
    print("Mean relative error: %e" % MEAN_RELATIVE_ERROR)
    print("Max relative error: %e" % MAX_RELATIVE_ERROR)

    # Expect both plots to appear identical:
    multi_plot(SYS.domain.t, [NO_OCL_POWER, OCL_POWER], z_labels=['CPU','GPU'],
                x_label=labels["t"], y_label=labels["P_t"], use_fill=False)

    print("*** Test of the Raman response ***")
    # -------------------------------------------------------------------------
    # Dudley_SC, fig 6
    TS = 2**13
    GAMMA = 110.0
    BETA = [0.0, 0.0, -11.830]
    STEPS = 800
    LENGTH = 50*1e-5
    WIDTH = 0.02836

    N_sol = 3.
    L_d = (WIDTH**2)/abs(BETA[2])
    #L_nl = 1/(gamma*P_0)
    L_nl = L_d/(N_sol**2)
    P_0 = 1/(GAMMA*L_nl)

    DOMAIN = Domain(bit_width=10.0, samples_per_bit=TS, centre_nu=lambda_to_nu(835.0))

    SYS = System(DOMAIN)
    SYS.add(Sech(peak_power=P_0, width=WIDTH))
    SYS.add(Fibre("fibre", beta=BETA, gamma=GAMMA, self_steepening=False, use_all='hollenbeck',
                  length=LENGTH, total_steps=STEPS, method="RK4IP"))

    start = time.time()
    SYS.run()
    stop = time.time()
    NO_OCL_DURATION = (stop - start)
    NO_OCL_OUT = SYS.fields["fibre"]

    sys = System(DOMAIN)
    sys.add(Sech(peak_power=P_0, width=WIDTH))
    sys.add(OpenclFibre("ocl_fibre", beta=BETA, gamma=GAMMA, self_steepening=False, use_all='hollenbeck',
                        length=LENGTH, total_steps=STEPS))

    start = time.time()
    sys.run()
    stop = time.time()
    OCL_DURATION = (stop - start)
    OCL_OUT = sys.fields["ocl_fibre"]

    NO_OCL_POWER = temporal_power(NO_OCL_OUT)
    OCL_POWER = temporal_power(OCL_OUT)
    DELTA_POWER = NO_OCL_POWER - OCL_POWER

    MEAN_RELATIVE_ERROR = np.mean(np.abs(DELTA_POWER))
    MEAN_RELATIVE_ERROR /= np.max(temporal_power(NO_OCL_OUT))
    
    MAX_RELATIVE_ERROR = np.max(np.abs(DELTA_POWER))
    MAX_RELATIVE_ERROR /= np.max(temporal_power(NO_OCL_OUT))

    print("Run time without OpenCL: %e" % NO_OCL_DURATION)
    print("Run time with OpenCL: %e" % OCL_DURATION)
    print("Mean relative error: %e" % MEAN_RELATIVE_ERROR)
    print("Max relative error: %e" % MAX_RELATIVE_ERROR)

    # Expect both plots to appear identical:
    multi_plot(SYS.domain.t, [NO_OCL_POWER, OCL_POWER], z_labels=['CPU','GPU'],
                x_label=labels["t"], y_label=labels["P_t"], use_fill=False)

    print("*** Test of the self-steepening contribution ***")
    # -------------------------------------------------------------------------
    # Dudley_SC, fig 8
    TS = 2**13
    GAMMA = 110.0
    BETA = [0.0, 0.0, -11.830, 
           8.1038e-2,  -9.5205e-5,   2.0737e-7,  
          -5.3943e-10,  1.3486e-12, -2.5495e-15, 
           3.0524e-18, -1.7140e-21]
    STEPS = 800
    LENGTH = 50*1e-5
    WIDTH = 0.010
    P_0 = 3480.
    
    try:
        clf = OpenclFibre("ocl_fibre", beta=BETA, gamma=GAMMA, self_steepening=True, use_all=False,
                        dorf="double", length=LENGTH, total_steps=STEPS)
    except NotImplementedError:
        print("Not Implemented, skipped")
    
    print("*** Test of the supercontinuum generation ***")
    # -------------------------------------------------------------------------
    # Dudley_SC, fig 3
    TS = 2**14
    GAMMA = 110.0
    BETA = [0.0, 0.0, -11.830, 
           8.1038e-2,  -9.5205e-5,   2.0737e-7,  
          -5.3943e-10,  1.3486e-12, -2.5495e-15, 
           3.0524e-18, -1.7140e-21]
    STEPS = 10000
    LENGTH = 15*1e-5
    WIDTH = 0.050
    P_0 = 10000.
    TAU_SHOCK = 0.56e-3

    DOMAIN = Domain(bit_width=10.0, samples_per_bit=TS, centre_nu=lambda_to_nu(835.0))

    SYS = System(DOMAIN)
    SYS.add(Sech(peak_power=P_0, width=WIDTH, using_fwhm=True))
    SYS.add(Fibre("fibre", beta=BETA, gamma=GAMMA, self_steepening=TAU_SHOCK, use_all='hollenbeck',
                  length=LENGTH, total_steps=STEPS, method="RK4IP"))

    start = time.time()
    SYS.run()
    stop = time.time()
    NO_OCL_DURATION = (stop - start)
    NO_OCL_OUT = SYS.fields["fibre"]

    sys = System(DOMAIN)
    sys.add(Sech(peak_power=P_0, width=WIDTH, using_fwhm=True))
    sys.add(OpenclFibre("ocl_fibre", beta=BETA, gamma=GAMMA, self_steepening=TAU_SHOCK, use_all='hollenbeck',
                        length=LENGTH, total_steps=STEPS))

    start = time.time()
    sys.run()
    stop = time.time()
    OCL_DURATION = (stop - start)
    OCL_OUT = sys.fields["ocl_fibre"]
    
    sys = System(DOMAIN)
    sys.add(Sech(peak_power=P_0, width=WIDTH, using_fwhm=True))
    sys.add(OpenclFibre("ocl_fibre", beta=BETA, gamma=GAMMA, self_steepening=TAU_SHOCK, use_all='hollenbeck',
                        method='acl_rk4ip', length=LENGTH))

    start = time.time()
    sys.run()
    stop = time.time()
    OCLA_DURATION = (stop - start)
    OCLA_OUT = sys.fields["ocl_fibre"]

    NO_OCL_POWER = temporal_power(NO_OCL_OUT)
    OCL_POWER = temporal_power(OCL_OUT)
    OCLA_POWER = temporal_power(OCLA_OUT)
    DELTA_POWER = NO_OCL_POWER - OCL_POWER

    MEAN_RELATIVE_ERROR = np.mean(np.abs(DELTA_POWER))
    MEAN_RELATIVE_ERROR /= np.max(temporal_power(NO_OCL_OUT))
    
    MAX_RELATIVE_ERROR = np.max(np.abs(DELTA_POWER))
    MAX_RELATIVE_ERROR /= np.max(temporal_power(NO_OCL_OUT))

    print("Run time without OpenCL: %e" % NO_OCL_DURATION)
    print("Run time with OpenCL: %e" % OCL_DURATION)
    print("Run time with OpenCL and adaptive step: %e" % OCLA_DURATION)
    print("Mean relative error: %e" % MEAN_RELATIVE_ERROR)
    print("Max relative error: %e" % MAX_RELATIVE_ERROR)

    # Expect both plots to appear identical:
    multi_plot(SYS.domain.t, [NO_OCL_POWER, OCL_POWER, OCLA_POWER], z_labels=['CPU','GPU','AGPU'],
                x_label=labels["t"], y_label=labels["P_t"], use_fill=False)


