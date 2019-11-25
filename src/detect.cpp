#include "detector.h"
#include <ctime>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <dlfcn.h>
#include "common.h"

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
        detector->Detect(raw_image, finalBoxInfos);

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
            detector->Detect(raw_image, finalBoxInfos);
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
