#include "opencl/test.h"



namespace opencl{
    namespace testing{
        template<>
            void InitRandomData<bool>(bool* data, size_t size){
                for(size_t i=0; i<size; ++i){
                    data[i] = random()%2==0;
                }
            }

    } // namespace testing
} // namespace opencl
