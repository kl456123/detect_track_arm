#include "opencl/ocl_driver.h"
#include <assert>


namespace opencl{
    namespace internal{
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
            IsInitialized = true;
            return CL_SUCCESS;
        }
    }// namespace

    OCLDriver::OCLDriver(){
    }

    Status OCLDriver::Init(){
        static Status* init_retval = [] {
            return new Status(InternalInit());
        }();
        return *init_retval;
    }

    void OCLDriver::IsInitialized(){
        return initialized;
    }

    /*static*/ Status OCLDriver::CreateContext(int device_ordinal,
            GpuDeviceHandle device, GpuContext* out){
        GpuStatus status;
        *out = clCreateContext(NULL, num_devices, devices.data(), NULL, NULL, &status);
        return status;
    }

    /*static*/ void OCLDriver::DestroyContext(GpuContext* context){
        clReleaseContext(*context)
    }

    /*static*/ bool OCLDriver::CreateStream(GpuContext* context,
            GpuStreamHandle* stream){
        if(clRetainContext(*context)!=CL_SUCCESS){
            return false;
        }

        GpuStatus status;
        // use the first device by default
        *stream = clCreateCommandQueueWithProperties(*context, devices[0], NULL, &status);
        if(status!=CL_SUCCESS){
            *stream = NULL;
            return false;
        }

        return true;
    }

    /*static*/ void* OCLDriver::DeviceAllocate(GpuContext* context, uint64 bytes){
        if(clRetainContext(*context)!=CL_SUCCESS){
            return NULL;
        }
        GpuStatus status;

        CUdeviceptr result = 0;
        result = clCreateBuffer(*context, CL_MEM_READ_ONLY, bytes, NULL, &status);
        void* ptr = reinterpret_cast<void*>(result);
        return ptr;
    }

    /*static*/ void OCLDriver::DeviceDeallocate(GpuContext* context, GpuDevicePtr gpu_ptr){
        if(clRetainContext(*context)!=CL_SUCCESS){
            return ;
        }
        clReleaseMemObject(gpu_ptr);
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

}// namespace opencl
