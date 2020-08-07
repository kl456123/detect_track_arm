#include "opencl/ocl_driver.h"
#include <iostream>
#include <glog/logging.h>
#include "opencl/gpu_kernel_helper.h"
#include "opencl/functors.h"

using namespace opencl;

int main(int argc, char* argv[]){
    FLAGS_log_dir = "./log.txt";
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr=1;
    FLAGS_minloglevel=0;
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

    auto a_ptr = OCLDriver::DeviceAllocate(ctx, bytes);
    auto b_ptr = OCLDriver::DeviceAllocate(ctx, bytes);
    auto c_ptr = OCLDriver::DeviceAllocate(ctx, bytes);

    // copy host to device
    OCLDriver::SynchronousMemcpyH2D(ctx, GpuDevicePtr(a_ptr), a, bytes);
    OCLDriver::SynchronousMemcpyH2D(ctx, GpuDevicePtr(b_ptr), b, bytes);

    // prepare kernel
    std::string fname = "../opencl/cl/vec_add.cl";
    const char* kernel_name = "vector_add";
    GpuFunctionHandle kernel;
    GpuModuleHandle program=0;
    if(OCLDriver::LoadPtx(ctx, fname, &program)!=CL_SUCCESS){
        LOG(FATAL)<<"Create Program Failed: "<<fname;
        return -1;
    }

    if(!OCLDriver::GetModuleFunction(ctx, program,
                kernel_name, &kernel)){
        LOG(FATAL)<<"Create Kernel Failed: "<<kernel_name;
        return -1;
    }

    // compute stream
    GpuStreamHandle stream;
    if(!OCLDriver::CreateStream(ctx, &stream)){
        LOG(FATAL)<<"Create Stream Failed: ";
        return -1;
    }

    GpuSetKernel(kernel, GpuDevicePtr(a_ptr), GpuDevicePtr(b_ptr), GpuDevicePtr(c_ptr));

    OCLDriver::LaunchKernel(ctx, kernel, num, 1,1/*gws*/,   1, 1, 1/*lws*/,
            0, stream, NULL, NULL);

    if(OCLDriver::SynchronousMemcpyD2H(ctx, c,
                GpuDevicePtr(c_ptr), bytes)!=CL_SUCCESS){
        LOG(FATAL)<<"Copy From Device To Host Failed";
        return -1;
    }

    OCLDriver::SynchronizeStream(ctx, stream);

    // print computed result
    for(int i=0; i<num; ++i){
        DLOG(INFO)<<c[i];
    }

    // clean
    OCLDriver::DestroyStream(ctx, stream);
    OCLDriver::DestroyContext(ctx);

    google::ShutdownGoogleLogging();
}
