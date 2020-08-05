#include "opencl/ocl_driver.h"
#include <iostream>

using namespace opencl;

int main(){
    // init platform and all devices
    OCLDriver::Init();

    GpuStatus status;
    GpuContext ctx;
    status = OCLDriver::CreateContext(0, &ctx);
    using dtype = float;
    const int num = 10;
    uint64 bytes = sizeof(dtype) * num;

    dtype* a = (dtype*)malloc(bytes);
    dtype* b = (dtype*)malloc(bytes);
    dtype* c = (dtype*)malloc(bytes);

    auto a_ptr = OCLDriver::DeviceAllocate(ctx, num);
    auto b_ptr = OCLDriver::DeviceAllocate(ctx, num);
    auto c_ptr = OCLDriver::DeviceAllocate(ctx, num);

    // copy host to device
    OCLDriver::SynchronousMemcpyH2D(ctx, GpuDevicePtr(a_ptr), a, bytes);
    OCLDriver::SynchronousMemcpyH2D(ctx, GpuDevicePtr(b_ptr), b, bytes);

    // prepare kernel
    const char* src_code = "";
    const char* kernel_name = "vec_add";
    GpuFunctionHandle kernel;
    GpuModuleHandle program=0;
    status = OCLDriver::LoadPtx(ctx, src_code, &program);

    if(!OCLDriver::GetModuleFunction(ctx, program,
                kernel_name, &kernel)){
        return -1;
    }

    // compute stream
    GpuStreamHandle stream;
    if(!OCLDriver::CreateStream(ctx, &stream)){
        return -1;
    }

    // Set the arguments of the kernel
    status = clSetKernelArg(kernel, 0, sizeof(GpuDevicePtr), (void *)&a_ptr);
    status = clSetKernelArg(kernel, 1, sizeof(GpuDevicePtr), (void *)&b_ptr);
    status = clSetKernelArg(kernel, 2, sizeof(GpuDevicePtr), (void *)&c_ptr);

    OCLDriver::LaunchKernel(ctx, kernel, num, 1,1/*gws*/,   1, 1, 1/*lws*/,
            0, stream, NULL, NULL);

    OCLDriver::SynchronousMemcpyD2H(ctx, c, GpuDevicePtr(c_ptr), bytes);

    // print computed result
    for(int i=0; i<num; ++i){
        std::cout<<c[i]<<std::endl;
    }
}
