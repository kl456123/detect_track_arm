#include "opencl/functors.h"
#include "opencl/ocl_driver.h"
#include <glog/logging.h>
#include "opencl/gpu_kernel_helper.h"

namespace opencl{
    namespace{
        // cache here
        void* input_ptr=nullptr;
        void* output_ptr=nullptr;
    } // namespace internal

    namespace functor{
        void MaxPool2D::operator()(const DeviceContext* device_context, const float* input,
                bool* output, const std::vector<int>& input_shape, const std::vector<int>& output_shape,
                int kernel_size, int stride_size){
            auto ctx = device_context->context;
            // calc total bytes used
            size_t num = 1;
            for(auto item:output_shape){
                num*=item;
            }
            uint64 input_bytes = num*sizeof(float);
            uint64 output_bytes = num*sizeof(bool);

            // allocate gpu mem
            if(!input_ptr) input_ptr = OCLDriver::DeviceAllocate(ctx, input_bytes);
            if(!output_ptr) output_ptr = OCLDriver::DeviceAllocate(ctx, output_bytes);

            // upload data
            OCLDriver::SynchronousMemcpyH2D(ctx, GpuDevicePtr(input_ptr), input, input_bytes);

            // prepare kernel
            std::string fname = "/home/breakpoint/Documents/detect_track_arm/src/opencl/opencl/cl/max_pool.cl";
            const char* kernel_name = "max_pool_2d";
            GpuFunctionHandle kernel;
            GpuModuleHandle program=0;

            CHECK(OCLDriver::LoadPtx(ctx, fname, &program))
                <<"Create Program Failed: "<<fname;

            CHECK(OCLDriver::GetModuleFunction(ctx, program,
                        kernel_name, &kernel))
                <<"Create Kernel Failed: "<<kernel_name;

            // compute stream
            GpuStreamHandle stream = device_context->stream;
            CHECK_NOTNULL(stream);

            GpuSetKernel(kernel, GpuDevicePtr(input_ptr), GpuDevicePtr(output_ptr),
                    kernel_size, stride_size);
            // note that sizeof(cl_int3)==16 instead of 12
            // TODO(breakpoint) set kernel arg more elegant
            GpuStatus ret;
            cl_int3 in_shape = {input_shape[0], input_shape[1], input_shape[2]};
            ret = clSetKernelArg(kernel, 4, sizeof(cl_int3), &in_shape);
            CHECK_EQ(ret, CL_SUCCESS)<<"set kernel arg in index error code: "<<ret;

            cl_int3 out_shape = {output_shape[0], output_shape[1], output_shape[2]};
            ret = clSetKernelArg(kernel, 5, sizeof(cl_int3), &out_shape);
            CHECK_EQ(ret, CL_SUCCESS)<<"set kernel arg in index error code: "<<ret;

            CHECK(OCLDriver::LaunchKernel(ctx, kernel, output_shape[0], output_shape[1],
                        output_shape[2]/*gws*/,   1, 1, 1/*lws*/,
                        0, stream, NULL, NULL))
                <<"Launch Kernel Failed";

            CHECK(OCLDriver::SynchronousMemcpyD2H(ctx, output,
                        GpuDevicePtr(output_ptr), output_bytes))
                <<"Copy From Device To Host Failed";
        }
    } // namespace functor
} // namespace opencl
