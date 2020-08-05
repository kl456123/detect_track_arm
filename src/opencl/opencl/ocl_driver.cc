#include <assert.h>
#include <fstream>
#include <vector>

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
            assert(platforms.size()>0);

            // copy to platforms
            platforms.reserve(num_platforms);
            platforms.assign(platforms_list, platforms_list+num_platforms);

            // get all devices
            clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
            GpuDeviceHandle devices_list[num_devices];
            clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, num_devices, devices_list, NULL);

            assert(devices.size()>0);
            initialized = true;
            return CL_SUCCESS;
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
        assert(devices.size()<device_ordinal);
        return CreateContext(devices[device_ordinal], context);
    }

    /*static*/ Status OCLDriver::CreateContext(GpuDeviceHandle device, GpuContext* out){
        GpuStatus status;
        *out = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
        return status;
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
            return CL_DEVICE_NOT_AVAILABLE;
        }

        *device = devices[device_ordinal];
        return CL_SUCCESS;
    }

    /*static*/ Status OCLDriver::GetDeviceName(GpuDeviceHandle device,
            string* device_name){

        return CL_SUCCESS;
    }

    /*static*/ Status OCLDriver::SynchronousMemcpyD2H(const GpuContext& context, void* host_dst,
            GpuDevicePtr gpu_src, uint64 size){
        GpuStreamHandle default_stream=0;
        auto ret = GetDefaultStream(context, &default_stream);
        GpuStatus status = clEnqueueReadBuffer(default_stream, gpu_src, CL_TRUE, 0, size, host_dst,
                0/*num of wait events*/, NULL/*wait events*/, NULL/*event*/);
        return status;
    }

    /*static*/ Status OCLDriver::SynchronousMemcpyH2D(const GpuContext& context,
            GpuDevicePtr gpu_dst,
            const void* host_src, uint64 size){
        GpuStreamHandle default_stream=0;
        auto ret = GetDefaultStream(context, &default_stream);
        GpuStatus status = clEnqueueWriteBuffer(default_stream, gpu_dst, CL_TRUE, 0, size, host_src,
                0, NULL, NULL);
        return status;
    }

    /*static*/ Status OCLDriver::SynchronousMemcpyD2D(const GpuContext& context,
            GpuDevicePtr gpu_dst,
            GpuDevicePtr gpu_src, uint64 size){
        GpuStreamHandle default_stream=0;
        auto ret = GetDefaultStream(context, &default_stream);
        if(CL_SUCCESS==clEnqueueCopyBuffer(default_stream, gpu_src,
                    gpu_dst, 0, 0, size, 0, NULL, NULL)){
            return true;
        }
        return false;
    }

    /*static*/ Status OCLDriver::GetDefaultStream(const GpuContext& context, GpuStreamHandle* stream){
        if(default_stream()==nullptr){
            CreateStream(context, &default_stream_);
        }

        *stream = default_stream_;
        return CL_SUCCESS;
    }

    /*static*/ bool OCLDriver::AsynchronousMemcpyD2H(const GpuContext& context, void* host_dst,
            GpuDevicePtr gpu_src, uint64 size,
            GpuStreamHandle stream){
    }

    /*static*/ bool OCLDriver::AsynchronousMemcpyH2D(const GpuContext& context, GpuDevicePtr gpu_dst,
            const void* host_src, uint64 size,
            GpuStreamHandle stream){
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
        return status;
    }

    /*static*/ Status OCLDriver::LoadPtx(const GpuContext& context, const std::string& fname,
            GpuModuleHandle* module){
        std::ifstream source_file(std::string(fname.data(), fname.size()));
        if(source_file.fail()){
            return CL_INVALID_PROGRAM;
        }
        std::string source_code(
                std::istreambuf_iterator<char>(source_file),
                (std::istreambuf_iterator<char>()));
        const char* c_str = source_code.c_str();
        const size_t str_size = source_code.size();

        GpuStatus ret;
        // Create a program from the kernel source
        GpuModuleHandle program = clCreateProgramWithSource(context, 1,
                &c_str, &str_size, &ret);

        // Build the program
        ret = clBuildProgram(program, 1, &devices[0], NULL, NULL, NULL);

        *module = program;
        return CL_SUCCESS;
    }

    /* static */ bool OCLDriver::GetModuleFunction(const GpuContext& context,
            GpuModuleHandle module, const char* kernel_name, GpuFunctionHandle* function) {
        if(clRetainContext(context)!=CL_SUCCESS){
            return false;
        }
        assert(module != nullptr && kernel_name != nullptr);
        // CHECK(module != nullptr && kernel_name != nullptr);
        GpuStatus status;
        GpuFunctionHandle kernel = clCreateKernel(module, kernel_name, &status);
        if (status != CL_SUCCESS) {
            return false;
        }

        return true;
    }

}// namespace opencl
