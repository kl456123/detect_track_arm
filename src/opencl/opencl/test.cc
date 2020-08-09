#include "opencl/test.h"



namespace opencl{
    namespace testing{
        void CheckTheSameCPU(const float* data_ptr1,
                const float* data_ptr2, size_t size){
            for(auto i=0; i<size;++i){
                EXPECT_EQ(data_ptr1[i], data_ptr2[i]);
            }
        }

        template<>
            void InitRandomData<bool>(bool* data, size_t size){
                for(size_t i=0; i<size; ++i){
                    data[i] = random()%2==0;
                }
            }

    } // namespace testing
} // namespace opencl
