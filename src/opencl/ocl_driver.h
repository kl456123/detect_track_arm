#ifndef OPENCL_OCL_DRIVER_H_
#define OPENCL_OCL_DRIVER_H_
#include "opencl/gpu_types.h"

namespace opencl{
    class OCLDriver{
        public:
            static Status Init();

            // Allocates a GPU memory space of size bytes associated with the given
            // context via cuMemAlloc.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467
            static void* DeviceAllocate(GpuContext* context, uint64 bytes);

            // Deallocates a GPU memory space of size bytes associated with the given
            // context via cuMemFree.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a
            static void DeviceDeallocate(GpuContext* context, void* location);

            static bool IsInitialized();

            // Given a device to create a context for, returns a context handle into the
            // context outparam, which must not be null.
            //
            // N.B. CUDA contexts are weird. They are implicitly associated with the
            // calling thread. Current documentation on contexts and their influence on
            // userspace processes is given here:
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g65dc0012348bc84810e2103a40d8e2cf
            static port::Status CreateContext(int device_ordinal, GpuDeviceHandle device,
                    const DeviceOptions& device_options,
                    GpuContext* context);

            // Destroys the provided context via cuCtxDestroy.
            // Don't do this while clients could still be using the context, per the docs
            // bad things will happen.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e
            static void DestroyContext(GpuContext* context);

            // Creates a new CUDA stream associated with the given context via
            // cuStreamCreate.
            // stream is an outparam owned by the caller, must not be null.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4
            static bool CreateStream(GpuContext* context, GpuStreamHandle* stream);

            // Destroys a CUDA stream associated with the given context.
            // stream is owned by the caller, must not be null, and *stream is set to null
            // if the stream is successfully destroyed.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758
            static void DestroyStream(GpuContext* context, GpuStreamHandle* stream);

            // Given a device ordinal, returns a device handle into the device outparam,
            // which must not be null.
            //
            // N.B. these device handles do not have a corresponding destroy function in
            // the CUDA driver API.
            static Status GetDevice(int device_ordinal, GpuDeviceHandle* device);

            // Given a device handle, returns the name reported by the driver for the
            // device.
            static Status GetDeviceName(GpuDeviceHandle device,
                    string* device_name);

            // -- Context- and device-independent calls.

            // Returns the number of visible CUDA device via cuDeviceGetCount.
            // This should correspond to the set of device ordinals available.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74
            static int GetDeviceCount();

            // Returns the driver version number via cuDriverGetVersion.
            // This is, surprisingly, NOT the actual driver version (e.g. 331.79) but,
            // instead, the CUDA toolkit release number that this driver is compatible
            // with; e.g. 6000 (for a CUDA 6.0 compatible driver) or 6050 (for a CUDA 6.5
            // compatible driver).
            //
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VERSION.html#group__CUDA__VERSION_1g8b7a10395392e049006e61bcdc8ebe71
            static bool GetDriverVersion(int* driver_version);

            // Returns the total amount of memory available for allocation by the CUDA
            // context, in bytes, via cuDeviceTotalMem.
            static bool GetDeviceTotalMemory(GpuDeviceHandle device, uint64* result);

            // Returns the free amount of memory and total amount of memory, as reported
            // by cuMemGetInfo.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0
            static bool GetDeviceMemoryInfo(GpuContext* context, int64* free,
                    int64* total);

            // Blocks the calling thread until the operations enqueued onto stream have
            // been completed, via cuStreamSynchronize.
            //
            // TODO(leary) if a pathological thread enqueues operations onto the stream
            // while another thread blocks like this, can you wind up waiting an unbounded
            // amount of time?
            //
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad
            static Status SynchronizeStream(GpuContext* context,
                    GpuStreamHandle stream);

            // Blocks the calling thread until the operations associated with the context
            // have been completed, via cuCtxSynchronize.
            //
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616
            static bool SynchronizeContext(GpuContext* context);

            // Returns true if all stream tasks have completed at time of the call. Note
            // the potential for races around this call (if another thread adds work to
            // the stream immediately after this returns).
            static bool IsStreamIdle(GpuContext* context, GpuStreamHandle stream);

            // -- Synchronous memcopies.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169

            static Status SynchronousMemcpyD2H(GpuContext* context, void* host_dst,
                    GpuDevicePtr gpu_src, uint64 size);
            static Status SynchronousMemcpyH2D(GpuContext* context,
                    GpuDevicePtr gpu_dst,
                    const void* host_src, uint64 size);
            static Status SynchronousMemcpyD2D(GpuContext* context,
                    GpuDevicePtr gpu_dst,
                    GpuDevicePtr gpu_src, uint64 size);

            // -- Asynchronous memcopies.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362

            static bool AsynchronousMemcpyD2H(GpuContext* context, void* host_dst,
                    GpuDevicePtr gpu_src, uint64 size,
                    GpuStreamHandle stream);
            static bool AsynchronousMemcpyH2D(GpuContext* context, GpuDevicePtr gpu_dst,
                    const void* host_src, uint64 size,
                    GpuStreamHandle stream);
            static bool AsynchronousMemcpyD2D(GpuContext* context, GpuDevicePtr gpu_dst,
                    GpuDevicePtr gpu_src, uint64 size,
                    GpuStreamHandle stream);
    };
}

#endif
