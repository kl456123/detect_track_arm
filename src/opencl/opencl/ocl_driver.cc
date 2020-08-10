#include <fstream>
#include <vector>
#include <glog/logging.h>

#include "opencl/ocl_driver.h"


namespace opencl{
    namespace {
        cl_uint num_platforms;
        std::vector<GpuPlatform> platforms;

        cl_uint num_devices;
        std::vector<GpuDeviceHandle> devices;
        bool initialized = false;
    }//namespace internal

    namespace{

        static Status InternalInit(){
            // create platform
            GpuStatus status = clGetPlatformIDs(0, NULL, &num_platforms);
            GpuPlatform platforms_list[num_platforms];
            clGetPlatformIDs(num_platforms, platforms_list, NULL);

            // copy to platforms
            platforms.reserve(num_platforms);
            platforms.assign(platforms_list, platforms_list+num_platforms);
            CHECK_GT(platforms.size(), 0);

            // get all devices
            clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
            GpuDeviceHandle devices_list[num_devices];
            clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, num_devices, devices_list, NULL);

            // copy to devices
            devices.reserve(num_devices);
            devices.assign(devices_list, devices_list + num_devices);

            CHECK_GT(devices.size(), 0);
            initialized = true;
            return CL_SUCCESS;
        }

        // get program build info
        string GetProgramBuildInfo(const GpuModuleHandle& program){
            // char *buff_erro;
            GpuStatus errcode;
            size_t build_log_len;
            errcode = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
            CHECK_EQ(errcode, CL_SUCCESS)<<"clGetProgramBuildInfo failed";

            char buff_erro[build_log_len];

            errcode = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, build_log_len, buff_erro, NULL);
            CHECK_EQ(errcode, CL_SUCCESS)<<"clGetProgramBuildInfo failed";

            return string(buff_erro);
        }

        // get kernel build info
        string GetKernelBuildInfo(){
        }

        // get launch kernel info
        string GetKernelLaunchInfo(){
        }
    }// namespace

    GpuStreamHandle OCLDriver::default_stream_;

    Status OCLDriver::Init(){
        static Status* init_retval = [] {
            return new Status(InternalInit());
        }();
        return *init_retval;
    }

    /*static*/ bool OCLDriver::IsInitialized(){
        return initialized;
    }

    Status OCLDriver::CreateContext(int device_ordinal, GpuContext* context){
        CHECK_GT(devices.size(), device_ordinal);
        return CreateContext(devices[device_ordinal], context);
    }

    /*static*/ Status OCLDriver::CreateContext(GpuDeviceHandle device, GpuContext* out){
        GpuStatus status;
        *out = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
        CHECK_EQ(status, CL_SUCCESS)<<"Failed to create context, error_code: "<<status;
        return true;
    }

    /*static*/ void OCLDriver::DestroyContext(const GpuContext& context){
        clReleaseContext(context);
    }

    /*static*/ bool OCLDriver::CreateStream(const GpuContext& context,
            GpuStreamHandle* stream){
        if(clRetainContext(context)!=CL_SUCCESS){
            return false;
        }

        GpuStatus status;
        // use the first device by default
        *stream = clCreateCommandQueueWithProperties(context, devices[0], NULL, &status);
        if(status!=CL_SUCCESS){
            *stream = NULL;
            return false;
        }

        return true;
    }

    /*static*/ void* OCLDriver::DeviceAllocate(const GpuContext& context, uint64 bytes){
        if(clRetainContext(context)!=CL_SUCCESS){
            return NULL;
        }
        GpuStatus status;

        GpuDevicePtr result = 0;
        result = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &status);
        CHECK_EQ(status, CL_SUCCESS)<<"Failed to Create Buffer, error code: "<<status;
        void* ptr = reinterpret_cast<void*>(result);
        return ptr;
    }

    /*static*/ void OCLDriver::DeviceDeallocate(const GpuContext& context, void* gpu_ptr){
        if(clRetainContext(context)!=CL_SUCCESS){
            return;
        }
        GpuDevicePtr pointer = reinterpret_cast<GpuDevicePtr>(gpu_ptr);
        if(clReleaseMemObject(pointer)!=CL_SUCCESS){
            abort();
        }
    }

    /*static*/ Status OCLDriver::GetDevice(int device_ordinal, GpuDeviceHandle* device){
        if(devices.size()<device_ordinal){
            LOG(FATAL)<<"illegal device ordinal error code: "<<device_ordinal;
        }

        *device = devices[device_ordinal];
        return true;
    }

    /*static*/ Status OCLDriver::GetDeviceName(GpuDeviceHandle device,
            string* device_name){

        return true;
    }

    /*static*/ Status OCLDriver::SynchronousMemcpyD2H(const GpuContext& context, void* host_dst,
            GpuDevicePtr gpu_src, uint64 size){
        GpuStreamHandle default_stream=0;
        CHECK(GetDefaultStream(context, &default_stream))<<"Create Default Stream Failed";
        GpuStatus status = clEnqueueReadBuffer(default_stream, gpu_src, CL_TRUE, 0, size, host_dst,
                0/*num of wait events*/, NULL/*wait events*/, NULL/*event*/);
        CHECK_EQ(status, CL_SUCCESS)<<"Failed to Read Buffer, Error Code: "<<status;
        return true;
    }

    /*static*/ Status OCLDriver::SynchronousMemcpyH2D(const GpuContext& context,
            GpuDevicePtr gpu_dst,
            const void* host_src, uint64 size){
        GpuStreamHandle default_stream=0;
        CHECK(GetDefaultStream(context, &default_stream))<<"Create Default Stream Failed";
        GpuStatus status = clEnqueueWriteBuffer(default_stream, gpu_dst, CL_TRUE, 0, size, host_src,
                0, NULL, NULL);
        CHECK_EQ(status, CL_SUCCESS)<<"Failed to Write Buffer error code: "<<status;
        return true;
    }

    /*static*/ Status OCLDriver::SynchronousMemcpyD2D(const GpuContext& context,
            GpuDevicePtr gpu_dst,
            GpuDevicePtr gpu_src, uint64 size){
        GpuStreamHandle default_stream=0;
        CHECK(GetDefaultStream(context, &default_stream))<<"Create Default Stream Failed";
        if(CL_SUCCESS==clEnqueueCopyBuffer(default_stream, gpu_src,
                    gpu_dst, 0, 0, size, 0, NULL, NULL)){
            return true;
        }
        return false;
    }

    /*static*/ Status OCLDriver::GetDefaultStream(const GpuContext& context, GpuStreamHandle* stream){
        if(default_stream()==nullptr){
            if(!CreateStream(context, &default_stream_)){
                LOG(FATAL)<<"create stream failed";
            }
        }

        *stream = default_stream_;
        return true;
    }

    /*static*/ bool OCLDriver::AsynchronousMemcpyD2H(const GpuContext& context, void* host_dst,
            GpuDevicePtr gpu_src, uint64 size,
            GpuStreamHandle stream){
        return true;
    }

    /*static*/ bool OCLDriver::AsynchronousMemcpyH2D(const GpuContext& context, GpuDevicePtr gpu_dst,
            const void* host_src, uint64 size,
            GpuStreamHandle stream){
        return true;
    }

    /*static*/ bool OCLDriver::AsynchronousMemcpyD2D(const GpuContext& context, GpuDevicePtr gpu_dst,
            GpuDevicePtr gpu_src, uint64 size,
            GpuStreamHandle stream){
        if(CL_SUCCESS==clEnqueueCopyBuffer(stream, gpu_src,
                    gpu_dst, 0, 0, size, 0, NULL, NULL)){
            return true;
        }
        return false;
    }

    /*static*/ Status OCLDriver::LaunchKernel(
            const GpuContext& context, GpuFunctionHandle function, unsigned int grid_dim_x,
            unsigned int grid_dim_y, unsigned int grid_dim_z,
            unsigned int block_dim_x, unsigned int block_dim_y,
            unsigned int block_dim_z, unsigned int shared_mem_bytes,
            GpuStreamHandle stream, void** kernel_params, void** extra){
        // enqueue
        const cl_uint work_dim = 3;
        const size_t gws[3] = {grid_dim_x, grid_dim_y, grid_dim_z};
        const size_t lws[3] = {block_dim_x, block_dim_y, block_dim_z};
        GpuStatus status = clEnqueueNDRangeKernel(stream, function, work_dim, NULL,
                gws, lws, 0, NULL, NULL);
        CHECK_EQ(status, CL_SUCCESS)<<" Failed to launch, error code: "<<status;
        return true;
    }

    /*static*/ Status OCLDriver::LoadPtx(const GpuContext& context, const std::string& fname,
            GpuModuleHandle* module){
        std::ifstream source_file(std::string(fname.data(), fname.size()));
        CHECK(!source_file.fail())<<"fname: "<<fname<<" Not Found!";

        std::string source_code(
                std::istreambuf_iterator<char>(source_file),
                (std::istreambuf_iterator<char>()));
        const char* c_str = source_code.c_str();
        const size_t str_size = source_code.size();

        GpuStatus ret;
        // Create a program from the kernel source
        GpuModuleHandle program = clCreateProgramWithSource(context, 1,
                &c_str, &str_size, &ret);
        CHECK_EQ(ret, CL_SUCCESS)<<"Create Program Failed, error code: "<<ret;

        // Build the program
        ret = clBuildProgram(program, 1, &devices[0], NULL, NULL, NULL);
        CHECK_EQ(ret, CL_SUCCESS)<<"Build Program Failed, build log: "<<GetProgramBuildInfo(program);

        *module = program;
        return true;
    }

    /* static */ bool OCLDriver::GetModuleFunction(const GpuContext& context,
            GpuModuleHandle module, const char* kernel_name, GpuFunctionHandle* function) {
        if(clRetainContext(context)!=CL_SUCCESS){
            return false;
        }
        CHECK(module != nullptr && kernel_name != nullptr);
        // CHECK(module != nullptr && kernel_name != nullptr);
        GpuStatus status;
        GpuFunctionHandle kernel = clCreateKernel(module, kernel_name, &status);
        if (status != CL_SUCCESS) {
            return false;
        }

        // set output
        *function = kernel;

        return true;
    }

    /* static */ Status OCLDriver::SynchronizeStream(const GpuContext& context,
            const GpuStreamHandle& stream) {
        CHECK_EQ(clRetainContext(context), CL_SUCCESS);
        clFlush(stream);
        clFinish(stream);
        return true;
    }

    /*static*/ void OCLDriver::DestroyStream(const GpuContext& context,
            const GpuStreamHandle& stream){
        clReleaseCommandQueue(stream);
    }

}// namespace opencl
