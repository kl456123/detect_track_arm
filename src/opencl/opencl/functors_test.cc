#include "opencl/functors.h"
#include "opencl/ocl_driver.h"
#include "opencl/gpu_types.h"
#include "opencl/test.h"


namespace opencl{
    namespace{
        /**
         * output_shape: (num_classes, height, width)
         */
        void MaxPool2DCPU(const float *input, bool *output, const std::vector<int>& input_shape,
                const std::vector<int>& output_shape,int stride_size, int kernel_size){
            for(int c=0; c<output_shape[0]; c++){
                for(int i=0; i<output_shape[1]; i++){
                    for(int j=0; j<output_shape[2]; j++){
                        const int output_index = (c*output_shape[1]+i)*output_shape[2]+j;
                        output[output_index] = true;
                        const int input_start_x = i*stride_size-1;
                        const int input_start_y = j*stride_size-1;
                        // value in the mid of pooled kernel
                        float mid_value = input[(c*input_shape[1]+input_start_x+kernel_size/2)*input_shape[2]
                            +input_start_y+kernel_size/2];
                        for(int k=0; k<kernel_size*kernel_size; ++k){
                            const int input_x = input_start_x + k/kernel_size;
                            const int input_y = input_start_y + k%kernel_size;
                            if(input_x<0||input_x>=input_shape[1]||input_y<0||input_y>=input_shape[2]){
                                continue;
                            }
                            const int input_index = (c*input_shape[1]+input_x)*input_shape[2]+input_y;
                            if(input[input_index]>mid_value){
                                output[output_index] = false;
                                continue;
                            }
                        }
                    }
                }
            }

        }
    } // namespace
    namespace{
        TEST(FunctorTest, MaxPool2DTest){
            // init ocl driver first
            OCLDriver::Init();

            // create ctx
            GpuStatus status;
            GpuContext ctx;
            EXPECT_TRUE(OCLDriver::CreateContext(0, &ctx));

            GpuStreamHandle stream;
            ASSERT_TRUE(OCLDriver::CreateStream(ctx, &stream));

            auto device_context = new DeviceContext;
            device_context->context = ctx;
            device_context->stream = stream;

            const int stride = 1;
            const int kernel_size = 3;
            const int num_channels = 1;
            const int height = 5;
            const int width = 5;
            const int num = height * width * num_channels;

            float input[num];
            bool output[num];
            bool output_cpu[num];
            testing::InitRandomData<float>(input, num);
            // testing::InitRandomData(output, num);
            // testing::InitRandomData(output_cpu, num);

            std::vector<int> output_shape = {num_channels, height, width};
            // stride = 1 so that input is the same size as output
            std::vector<int> input_shape = output_shape;
            functor::MaxPool2D()(device_context, input, output, input_shape, output_shape,
                    kernel_size, stride);

            MaxPool2DCPU(input, output_cpu, input_shape, output_shape,
                    kernel_size, stride);

            // check the result
            // testing::CheckTheSameCPU(output, output_cpu, num);
        }
    } // namespace
} // namespace opencl
