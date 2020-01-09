#include "centernet_detector.h"
#include <iterator>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/imgproc.hpp>


CenterNetDetector::CenterNetDetector(std::string& modelName, int width, int height, float nms_threshold,
        float score_threshold):Detector(modelName, width, height, nms_threshold, score_threshold){
}

void CenterNetDetector::PrepareInputAndOutputNames(){
    // too many output will crash the program
    // mOutputNames = {"hm", "wh", "reg"};
    mOutputNames =  {"total"};
}

void CenterNetDetector::Preprocess(const cv::Mat raw_image, cv::Mat& image){
    image = raw_image;

    mOriginInputSize.push_back(raw_image.rows);
    mOriginInputSize.push_back(raw_image.cols);
    cv::Mat inp_image = cv::Mat::zeros(mInputSize[0], mInputSize[1], image.type());

    cv::resize(image, inp_image, cv::Size(mInputSize[1], mInputSize[0]));

    inp_image.convertTo(inp_image, CV_32FC3);
    const float mean_vals[3] = { 0.408, 0.447, 0.47};
    const float std_vals[3] = { 0.289, 0.274, 0.278};

    std::vector<cv::Mat>mv;
    cv::split(inp_image, mv);
    for(int i=0;i<mv.size();i++){
        mv[i] = (mv[i]/255.f-mean_vals[i])/std_vals[i];
    }
    cv::merge(mv, image);
}









void CenterNetDetector::GenerateBoxInfo(std::vector<BoxInfo>& top_boxes, float score_threshold){
    auto tensors_host = mOutputTensorsHost;

    // auto hm_dataPtr  = tensors_host[0]->host<float>();
    // auto wh_dataPtr   = tensors_host[1]->host<float>();
    // auto reg_dataPtr = tensors_host[2]->host<float>();
    auto total_dataPtr = tensors_host[0]->host<float>();


    int raw_image_width = mOriginInputSize[1];
    int raw_image_height = mOriginInputSize[0];
    mNumOfClasses = tensors_host[0]->channel() - 4;

    auto spatial_dims = tensors_host[0]->shape();

    int num = tensors_host[0]->elementSize();

    // sigmoid for heatmap

    int spatial_size = spatial_dims[2]* spatial_dims[3];
    auto reg_dataPtr = total_dataPtr+mNumOfClasses*spatial_size;
    auto wh_dataPtr =  total_dataPtr+(mNumOfClasses+2)*spatial_size;

    for(int c=0;c<mNumOfClasses;c++){
        std::vector<BoxInfo> top_boxes_per_classes;
        for(int i=0;i<spatial_dims[2];i++){
            for(int j=0;j<spatial_dims[3];j++){
                float max=-1;
                int max_id = 0;
                for(int k=0;k<9;k++){
                    // get zero when index out of range
                    int tmp_i = i+k/3-1;
                    int tmp_j = j+k%3-1;
                    if (tmp_i<0 || tmp_i>=spatial_dims[2] || tmp_j<0 || tmp_j>=spatial_dims[3]){
                        continue;
                    }
                    int index = tmp_i*spatial_dims[3] + tmp_j;
                    float value = total_dataPtr[index];
                    if(max<value){
                        max = value;
                        max_id = k;
                    }
                }
                if(max_id!=4){
                    continue;
                }
                int index = (i+max_id/3-1)* spatial_dims[3] + (j+max_id%3-1);
                if(total_dataPtr[index]<score_threshold){
                    continue;
                }
                BoxInfo box_info;
                // box
                box_info.score = total_dataPtr[index];
                box_info.index = index;
                box_info.class_name = static_cast<CLASS_NAME>(c);

                top_boxes_per_classes.push_back(box_info);
            }
        }
        GetTopK(top_boxes_per_classes, mTopK);
        std::copy(top_boxes_per_classes.begin(), top_boxes_per_classes.end(), back_inserter(top_boxes));
        total_dataPtr+=spatial_size;
    }

    GetTopK(top_boxes, mTopK);

    float sx = 1.0*raw_image_width/spatial_dims[3];
    float sy = 1.0*raw_image_height/spatial_dims[2];

    for(auto&top_box:top_boxes){
        int index = top_box.index;
        int cx = index%spatial_dims[3];
        int cy = index/spatial_dims[3];
        top_box.cx = ((reg_dataPtr[index] + cx+0.5))*sx;
        top_box.cy = (reg_dataPtr[index+spatial_size] +cy+0.5)*sy;
        top_box.width = wh_dataPtr[index]*sx;
        top_box.height = wh_dataPtr[index+spatial_size]*sy;

        top_box.box.x = top_box.cx-0.5*top_box.width;
        top_box.box.y = top_box.cy-0.5*top_box.height;
        top_box.box.width = top_box.width;
        top_box.box.height = top_box.height;
    }
}


void CenterNetDetector::Detect(const cv::Mat& raw_image, std::vector<BoxInfo>& finalBoxInfos){
    // preprocess
    cv::Mat image;
    std::cout<<"Preprocessing "<<std::endl;
    Preprocess(raw_image, image);


    std::cout<<"Running "<<std::endl;
    Run(image);

    std::cout<<"Postprocessing "<<std::endl;
    // postprocess
    // std::vector<BoxInfo> boxInfos, boxInfos_left;
    GenerateBoxInfo(finalBoxInfos, mScoreThreshold);
    // top k
    // GetTopK(boxInfos, boxInfos_left, mTopK);
    // nms
    // NMS(boxInfos_left, finalBoxInfos, mNMSThreshold);

    // handle corner case
}
