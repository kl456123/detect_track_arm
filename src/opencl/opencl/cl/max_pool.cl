__kernel void max_pool_2d(__global float* input_image,
        __global bool* output_image,
        int kernel_size,
        int stride_size,
        int3 output_shape){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    int output_index = (i*output_shape.y+j)*output_shape.z+k;
    output_image[output_index] = 1;
    int half_kernel_size= kernel_size/2;
    // in the same position like output
    float mid_value = input_image[output_index];
    for(int x=-half_kernel_size;x<=half_kernel_size;++x){
        for(int y=-half_kernel_size;y<=half_kernel_size;++y){
            int input_index = (i*output_shape.y+j+x)*output_shape.z+k+y;
            if(input_image[input_index]>mid_value){
                output_image[output_index] = 0;
                // early stop
                return;
            }
        }
    }
}
