#include "common.h"

void drawBoxes(std::vector<BoxInfo>& finalBoxInfos, cv::Mat& raw_image){
    // visualize bbox
    for (auto& face: finalBoxInfos)
    {
        cv::Rect vis_box;
        vis_box.x = (int) face.box.x;
        vis_box.y = (int) face.box.y;
        vis_box.width  = (int) face.box.width;
        vis_box.height = (int) face.box.height;
        cv::rectangle(raw_image, vis_box, cv::Scalar(0,0,255), 1);
    }

}

void *open_video_stream(const std::string& f, int c, int w, int h, int fps)
{
    cv::VideoCapture *cap;
    if(!f.empty()) cap = new cv::VideoCapture(f.c_str());
    else cap = new cv::VideoCapture(c);
    if(!cap->isOpened()) return NULL;
    if(w) cap->set(CV_CAP_PROP_FRAME_WIDTH, w);
    if(h) cap->set(CV_CAP_PROP_FRAME_HEIGHT, h);
    if(fps) cap->set(CV_CAP_PROP_FPS, w);
    return cap;
}

void loadOpenCLLib(){
    auto handle = dlopen("/home/indemind/Documents/MNN/build/source/backend/opencl/libMNN_CL.so", RTLD_NOW);
    FUNC_PRINT_ALL(handle, p);
}

float iou(cv::Rect box0, cv::Rect box1)
{
    float xmin0 = box0.x;
    float ymin0 = box0.y;
    float xmax0 = box0.x + box0.width;
    float ymax0 = box0.y + box0.height;

    float xmin1 = box1.x;
    float ymin1 = box1.y;
    float xmax1 = box1.x + box1.width;
    float ymax1 = box1.y + box1.height;

    float w = fmax(0.0f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1));
    float h = fmax(0.0f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1));

    float i = w * h;
    float u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i;

    if (u <= 0.0) return 0.0f;
    else          return i/u;
}

float get_size(float w, float h){
    float pad  = (w+h)*0.5;
    return std::sqrt((w+pad)*(h+pad));
}
