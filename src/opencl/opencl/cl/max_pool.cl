__kernel void max_pool_2d(__global float* input_image,
                                   __global float* output_image
                                   __constant int kernel_size,
                                   __constant int stride_size,
                                   __constant int2 input_shape){
        int i = get_global_id(0);
        int j = get_global_id(1);
        int index = i*input_shape.x + j;
        for(int i=0;i<kernel_size;++i){
            if(kernel_size){
            }
        }
}
