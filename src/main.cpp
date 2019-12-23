#include "detector.h"
#include <ctime>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <dlfcn.h>

enum INPUTMODE{
    IMAGE=0,
    VIDEO=1,
};

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

void *open_video_stream(const char *f, int c, int w, int h, int fps)
{
    cv::VideoCapture *cap;
    if(f) cap = new cv::VideoCapture(f);
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


int main(int argc, char** argv){
    loadOpenCLLib();
    if(argc<2){
        std::cout<<"detector model_name image_name [mode]"<<std::endl;
        return -1;
    }

    std::string model_name(argv[1]);
    std::string image_or_video_file;
    if(argc>=3){
        image_or_video_file = argv[2];
    }
    INPUTMODE mode;
    if(argc>=4){
        mode = static_cast<INPUTMODE>(atoi(argv[3]));
    }else{
        mode = INPUTMODE::VIDEO;
    }

    auto detector = std::shared_ptr<Detector>(new Detector(model_name));

    std::vector<BoxInfo> finalBoxInfos;
    cv::Mat raw_image;
    if(mode==INPUTMODE::IMAGE){
        // read from image
        raw_image = cv::imread(image_or_video_file.c_str());
        detector->detect(raw_image, finalBoxInfos);

        drawBoxes(finalBoxInfos, raw_image);

        cv::imwrite("./output.jpg", raw_image);
    }else if(mode==INPUTMODE::VIDEO){
        // read from video or camera
        // cv::VideoCapture* cap(image_or_video_file);
        auto cap = std::shared_ptr<cv::VideoCapture>(reinterpret_cast<cv::VideoCapture*>(open_video_stream(0, -1, 320, 240, 0)));
        while (true) {
            finalBoxInfos.clear();
            *(cap.get()) >> raw_image;
            std::chrono::time_point<std::chrono::system_clock> t1 = std::chrono::system_clock::now();
            detector->detect(raw_image, finalBoxInfos);
            std::chrono::time_point<std::chrono::system_clock> t2 = std::chrono::system_clock::now();
            float dur = (float)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000;
            std::cout << "duration time:" << dur << "ms" << std::endl;

            drawBoxes(finalBoxInfos, raw_image);
            cv::namedWindow("MNN", CV_WINDOW_NORMAL);
            cv::imshow("MNN", raw_image);
            cv::waitKey(1);
        }
    }





    return 0;
}
