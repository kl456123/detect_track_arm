#ifndef OPENCL_GPU_KERNEL_HELPER_H_
#define OPENCL_GPU_KERNEL_HELPER_H_
#include "opencl/gpu_types.h"

namespace opencl{
    template<typename T>
        void KernelSetArgFromList(GpuFunctionHandle kernel, int i,
                T&& arg){
            CHECK_EQ(clSetKernelArg(kernel, i, sizeof(T), (void *)&arg), CL_SUCCESS);
        }
    template<typename FirstArg, typename ...RestArgs>
        void KernelSetArgFromList(GpuFunctionHandle kernel, int i, FirstArg&& first_arg, RestArgs&&... rest_args){
            CHECK_EQ(clSetKernelArg(kernel, i, sizeof(FirstArg), (void *)&first_arg), CL_SUCCESS);
            KernelSetArgFromList(kernel, i+1, rest_args...);
        }

    template<typename ...Args>
        Status GpuSetKernel(GpuFunctionHandle kernel, Args&&... args){
            KernelSetArgFromList(kernel, 0, args...);
            return CL_SUCCESS;
        }
} // namespace opencl


#endif
