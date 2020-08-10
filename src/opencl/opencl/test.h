#include <gtest/gtest.h>

#include <random>


namespace opencl{
    namespace testing{
        template<typename T>
            void CheckTheSameCPU(const T* data_ptr1,
                    const T* data_ptr2, size_t size){
                for(auto i=0; i<size;++i){
                    EXPECT_EQ(data_ptr1[i], data_ptr2[i]);
                }
            }


        // generate random data used for testing

        template<typename T>
            void InitRandomData(T* data, size_t size);


        template<typename T>
            void InitRandomData(T* data, size_t size){
                for(int i=0; i<size; ++i){
                    data[i] = T(1.0*random()/RAND_MAX);
                }
            }
    } // namespace testing
}
