#include <ctime>
#include <chrono>
#include <opencv2/core/core.hpp>
#include "tracker.h"
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

    auto tracker = std::shared_ptr<Tracker>(new Tracker(model_name));

    std::vector<BoxInfo> finalBoxInfos;
    cv::Mat raw_image;
    bool first_frame = true;

    auto cap = std::shared_ptr<cv::VideoCapture>(reinterpret_cast<cv::VideoCapture*>(open_video_stream(image_or_video_file, -1, 0, 0, 0)));
    while (true) {
        *(cap.get()) >> raw_image;
        if(raw_image.empty()){
            break;
        }
        if(first_frame){
            BoxInfo box_info;
            box_info.box = cv::selectROI("select", raw_image, false, false);
            cv::destroyWindow("select");
            tracker->Init(raw_image, box_info);
            first_frame = false;
        }else{
            finalBoxInfos.clear();
            std::chrono::time_point<std::chrono::system_clock> t1 = std::chrono::system_clock::now();
            tracker->Track(raw_image, finalBoxInfos);
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
