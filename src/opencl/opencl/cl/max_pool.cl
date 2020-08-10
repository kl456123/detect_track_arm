__kernel void max_pool_2d(__global float* input_image/*(num_classes, height, width)*/,
        __global bool* output_image,
        int kernel_size,
        int stride_size,
        int3 input_shape,
        int3 output_shape){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    int output_index = (i*output_shape.y+j)*output_shape.z+k;

    float mid_value = input_image[(i*input_shape.y+j*stride_size)
        *input_shape.z+k*stride_size];
    output_image[output_index] = true;
    int pad = kernel_size/2;
    for(int k_i=0; k_i<kernel_size*kernel_size; ++k_i){
        int input_x = j*stride_size+k_i/3-pad;
        int input_y = k*stride_size+k_i%3-pad;
        // ignore when out of bordary
        if(input_x<0||input_x>=input_shape.y){
            continue;
        }
        if(input_y<0||input_y>=input_shape.z){
            continue;
        }
        int input_index = (i*input_shape.y+input_x)
            *input_shape.z+input_y;
        if(input_image[input_index]>mid_value){
            output_image[output_index] = false;
            // early return
            return;
        }
        /* output_image[output_index] = max(output_image[output_index], input_image[input_index]); */
    }
}
