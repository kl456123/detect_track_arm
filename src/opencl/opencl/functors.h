#ifndef OPENCL_FUNCTORS_H_
#define OPENCL_FUNCTORS_H_
#include <vector>
#include "opencl/gpu_types.h"

namespace opencl{
    namespace functor{
        struct MaxPool2D{
            void operator()(const DeviceContext* ctx, const float* input,
                    bool* output, const std::vector<int>& input_shape,
                    const std::vector<int>& output_shape, int kernel_size,
                    int stride_size);
        };

        struct Add{
            void operator()(const DeviceContext* ctx, const float* input1,
                    const float* input2, float* output, int num);
        };
    } // namespace functor
}// namespace opencl


#endif
