
"""
    Copyright (C) 2013 David Bolt

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

import sys as sys0
version_py = sys0.version_info[0]
if version_py == 3:
    from reikna import cluda
    from reikna.fft import FFT
else:
    from pyfft.cl import Plan  
from string import Template

from .linearity import Linearity

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
""")


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
                 beta=None, gamma=0.0, method="cl_rk4ip", total_steps=100,
                 centre_omega=None, dorf='float', ctx=None, fast_math=False):

        self.name = name

        self.gamma = gamma
        
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
        self.buf_temp = None
        self.buf_interaction = None
        self.buf_factor = None

        self.shape = None
        self.plan = None

        self.cached_factor = False
        # Force usage of cached version of function:
        self.cl_linear = self.cl_linear_cached

        self.length = length
        self.total_steps = total_steps

        self.stepsize = self.length / self.total_steps
        self.zs = np.linspace(0.0, self.length, self.total_steps + 1)

        self.method = getattr(self, method.lower())
        
        self.linearity = Linearity(alpha, beta, sim_type="default",
                                    use_cache=True, centre_omega=centre_omega, phase_lim=True)
        self.factor = None

    def __call__(self, domain, field):
        # Setup plan for calculating fast Fourier transforms:
        if self.plan is None:
            if version_py == 3:
                self.plan = FFT(domain.t.astype(self.np_complex)).compile(self.thr, 
                        fast_math=self.fast_math, compiler_options=self.compiler_options)
                self.plan.execute = self.reikna_fft_execute
            else:
                self.plan = Plan(domain.total_samples, queue=self.queue, dtype=self.np_complex, fast_math=self.fast_math)

        if self.factor is None:
            self.factor = self.linearity(domain)

        self.send_arrays_to_device(field, self.factor)

        for z in self.zs[:-1]:
            self.method(self.buf_field, self.buf_temp,
                          self.buf_interaction, self.buf_factor, self.stepsize)

        return self.buf_field.get()

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
        self.shape = field.shape

        self.buf_field = cl_array.to_device(
            self.queue, field.astype(self.np_complex))

        if self.buf_temp is None:
            self.buf_temp = cl_array.empty_like(self.buf_field)
        if self.buf_interaction is None:
            self.buf_interaction = cl_array.empty_like(self.buf_field)

        if self.cached_factor is False:
            self.buf_factor = cl_array.to_device(
                self.queue, factor.astype(self.np_complex))

    def cl_copy(self, dst_buffer, src_buffer):
        """ Copy contents of one buffer into another. """
        cl.enqueue_copy(self.queue, dst_buffer.data, src_buffer.data).wait()

    def cl_linear(self, field_buffer, stepsize, factor_buffer):
        """ Linear part of step. """
        self.plan.execute(field_buffer.data, inverse=True)
        self.prg.cl_linear(self.queue, self.shape, None, field_buffer.data,
                           factor_buffer.data, self.np_float(stepsize))
        self.plan.execute(field_buffer.data)

    def cl_linear_cached(self, field_buffer, stepsize, factor_buffer):
        """ Linear part of step (cached version). """
        if (self.cached_factor is False):
            self.linearity.cache(stepsize)
            self.buf_factor = cl_array.to_device(
                self.queue, self.linearity.cached_factor.astype(self.np_complex))
            self.cached_factor = True

        self.plan.execute(field_buffer.data, inverse=True)
        self.prg.cl_linear_cached(self.queue, self.shape, None,
                                  field_buffer.data, self.buf_factor.data)
        self.plan.execute(field_buffer.data)

    def cl_n(self, field_buffer, stepsize):
        """ Nonlinear part of step. """
        self.prg.cl_nonlinear(self.queue, self.shape, None, field_buffer.data,
                              self.np_float(self.gamma), self.np_float(stepsize))
    
    def cl_nonlinear(self, fieldA_buffer, stepsize, fieldB_buffer):
        """ Nonlinear part of step, exponential term"""
        self.prg.cl_nonlinear_exp(self.queue, self.shape, None, fieldA_buffer.data, fieldB_buffer.data,
                              self.np_float(self.gamma), self.np_float(stepsize))

    def cl_sum(self, first_buffer, first_factor, second_buffer, second_factor):
        """ Calculate weighted summation. """
        self.prg.cl_sum(self.queue, self.shape, None,
                        first_buffer.data, self.np_float(first_factor),
                        second_buffer.data, self.np_float(second_factor))

    def reikna_fft_execute(self, d, inverse=False):
        self.plan(d,d,inverse=inverse)
    
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

        self.cl_linear(field, half_step, factor) #A_L

        self.cl_copy(field_temp, field)
        self.cl_copy(field_linear, field)
        self.cl_n(field_temp, stepsize) #k0
        self.cl_sum(field, 1, field_temp, inv_six) #free k0

        self.cl_sum(field_temp, 0.5, field_linear, 1)
        self.cl_n(field_temp, stepsize) #k1
        self.cl_sum(field, 1, field_temp, inv_three) #free k1
        
        self.cl_sum(field_temp, 0.5, field_linear, 1)
        self.cl_n(field_temp, stepsize) #k2
        self.cl_sum(field, 1, field_temp, inv_three) #free k2
        
        self.cl_sum(field_temp, 1, field_linear, 1)
        self.cl_n(field_temp, stepsize) #k3
        self.cl_sum(field, 1, field_temp, inv_six) #free k3
        
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

        self.cl_copy(field_interaction, field) #A_I
        self.cl_n(field_temp, stepsize)
        self.cl_linear(field_temp, half_step, factor) #k0

        self.cl_sum(field, 1.0, field_temp, inv_six) #free k0
        self.cl_sum(field_temp, 0.5, field_interaction, 1.0)
        self.cl_n(field_temp, stepsize) #k1

        self.cl_sum(field, 1.0, field_temp, inv_three) #free k1
        self.cl_sum(field_temp, 0.5, field_interaction, 1.0)
        self.cl_n(field_temp, stepsize) #k2

        self.cl_sum(field, 1.0, field_temp, inv_three) #free k2
        self.cl_sum(field_temp, 1.0, field_interaction, 1.0)
        self.cl_linear(field_temp, half_step, factor)
        self.cl_n(field_temp, stepsize) #k3

        self.cl_linear(field, half_step, factor)

        self.cl_sum(field, 1.0, field_temp, inv_six)

if __name__ == "__main__":
    # Compare simulations using Fibre and OpenclFibre modules.
    from pyofss import Domain, System, Gaussian, Fibre
    from pyofss import temporal_power, double_plot, labels, lambda_to_nu

    import time

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
                        dorf="float", length=LENGTH, total_steps=STEPS))

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
    double_plot(SYS.domain.t, NO_OCL_POWER, SYS.domain.t, OCL_POWER,
                x_label=labels["t"], y_label=labels["P_t"],
                X_label=labels["t"], Y_label=labels["P_t"])

