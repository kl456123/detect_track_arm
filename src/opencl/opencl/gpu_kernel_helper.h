#ifndef OPENCL_GPU_KERNEL_HELPER_H_
#define OPENCL_GPU_KERNEL_HELPER_H_
#include "opencl/gpu_types.h"

namespace opencl{
    template<typename T>
        void KernelSetArgFromList(GpuFunctionHandle kernel, int i,
                T&& arg){
            CHECK_EQ(clSetKernelArg(kernel, i, sizeof(T), (void *)&arg), CL_SUCCESS)
                <<"error in index of arg list: "<<i;
        }

    template<typename FirstArg, typename ...RestArgs>
        void KernelSetArgFromList(GpuFunctionHandle kernel, int i,
                FirstArg&& first_arg, RestArgs&&... rest_args){
            KernelSetArgFromList(kernel, i, first_arg);
            KernelSetArgFromList(kernel, i+1, rest_args...);
        }

    template<typename ...Args>
        Status GpuSetKernel(GpuFunctionHandle kernel, Args&&... args){
            KernelSetArgFromList(kernel, 0, args...);
            return true;
        }
} // namespace opencl


#endif
