#ifndef OPENCL_GPU_TYPES_H_
#define OPENCL_GPU_TYPES_H_
#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace opencl{
    // opencl types
    using GpuContext = cl_context;
    using GpuStreamHandle = cl_command_queue;
    using GpuFunctionHandle = cl_kernel;
    using GpuModuleHandle = cl_program;
    using GpuEventHandle = cl_event;
    using GpuDeviceHandle = cl_device_id;
    using GpuDevicePtr = cl_mem;
    using GpuStatus = cl_int;
    using GpuPlatform=cl_platform_id;

    // common types
    using Status=bool;
}// namespace




#endif
