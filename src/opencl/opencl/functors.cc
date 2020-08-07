#include "opencl/functors.h"
#include "opencl/ocl_driver.h"
#include <glog/logging.h>
#include "opencl/gpu_kernel_helper.h"

namespace opencl{
    namespace internal{
        void RunOpenCLProgram(){
        }
    } // namespace internal

    namespace functor{
        void MaxPool2D::operator()(const DeviceContext* device_context, const float* input,
                float* output, std::vector<int> output_shape,
                int kernel_size, int stride_size){
            auto ctx = device_context->context;
            // calc total bytes used
            size_t num = 1;
            for(auto item:output_shape){
                num*=item;
            }
            uint64 bytes = num*sizeof(float);

            // allocate gpu mem
            auto input_ptr = OCLDriver::DeviceAllocate(ctx, bytes);
            auto output_ptr = OCLDriver::DeviceAllocate(ctx, bytes);

            // upload data
            OCLDriver::SynchronousMemcpyH2D(ctx, GpuDevicePtr(input_ptr), input, bytes);

            // prepare kernel
            std::string fname = "/home/indemind/Documents/Project/src/opencl/opencl/cl/max_pool.cl";
            const char* kernel_name = "max_pool_2d";
            GpuFunctionHandle kernel;
            GpuModuleHandle program=0;
            if(OCLDriver::LoadPtx(ctx, fname, &program)!=CL_SUCCESS){
                LOG(FATAL)<<"Create Program Failed: "<<fname;
            }

            if(!OCLDriver::GetModuleFunction(ctx, program,
                        kernel_name, &kernel)){
                LOG(FATAL)<<"Create Kernel Failed: "<<kernel_name;
            }

            // compute stream
            GpuStreamHandle stream = device_context->stream;
            if(stream==NULL){
                LOG(FATAL)<<"Empty Stream";
            }


            GpuSetKernel(kernel, GpuDevicePtr(input_ptr), GpuDevicePtr(output_ptr),
                    output_shape.data(), 3, 1);

            OCLDriver::LaunchKernel(ctx, kernel, num, 1,1/*gws*/,   1, 1, 1/*lws*/,
                    0, stream, NULL, NULL);

            if(OCLDriver::SynchronousMemcpyD2H(ctx, output,
                        GpuDevicePtr(output_ptr), bytes)!=CL_SUCCESS){
                LOG(FATAL)<<"Copy From Device To Host Failed";
            }
        }
    } // namespace functor
} // namespace opencl
