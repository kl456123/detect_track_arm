#include "detector.h"
#include "centernet_detector.h"
#include <ctime>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <dlfcn.h>
#include "common.h"

#include <thread>

#include "instance_manager.h"

#ifdef USE_SDK
#include "sdk.h"

#endif




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

    std::string output_name;
    if(argc>=5){
        output_name = argv[4];
    }else{
        output_name = "./output.jpg";
    }

    std::shared_ptr<Detector> detector;
    std::cout<<model_name<<std::endl;
    detector.reset(new Detector(model_name));
    detector->InitInputAndOutput();
    // auto detector = std::shared_ptr<Detector>(new Detector(model_name));
    auto instance_manager = std::shared_ptr<InstanceManager>(new InstanceManager);
    std::vector<InstanceInfo> instance_infos;
#ifdef USE_SDK
    auto sdk =  std::shared_ptr<SDK>(new SDK());
    auto camera_param = sdk->GetModuleParams();
    sdk->PrintInfo();
    instance_manager->SetupCamera(camera_param);
#else
    instance_manager->SetupCamera();
#endif

    std::vector<BoxInfo> finalBoxInfos;
    cv::Mat raw_image;
    if(mode==INPUTMODE::IMAGE){
        // read from image
        raw_image = cv::imread(image_or_video_file.c_str());
        cv::cvtColor(raw_image, raw_image, CV_BGR2GRAY);
        cv::cvtColor(raw_image, raw_image, CV_GRAY2BGR);
        detector->Detect(raw_image, finalBoxInfos);

        drawBoxes(finalBoxInfos, raw_image);

        cv::imwrite(output_name, raw_image);
    }else if(mode==INPUTMODE::VIDEO){
        // read from video or camera
        // cv::VideoCapture* cap(image_or_video_file);
        auto cap = std::shared_ptr<cv::VideoCapture>(reinterpret_cast<cv::VideoCapture*>(open_video_stream(image_or_video_file, -1, 640, 480, 0)));
        while (true) {
            finalBoxInfos.clear();
            instance_infos.clear();
#ifndef USE_SDK
            *(cap.get()) >> raw_image;
            cv::cvtColor(raw_image, raw_image, CV_BGR2GRAY);
            cv::cvtColor(raw_image, raw_image, CV_GRAY2BGR);
#else
            raw_image = sdk->ReadImage();
            cv::cvtColor(raw_image, raw_image, CV_GRAY2BGR);
            auto pose = sdk->GetPose();
#endif
            // from single channel to 3 channels
            std::chrono::time_point<std::chrono::system_clock> t1 = std::chrono::system_clock::now();
            // first detect
            detector->Detect(raw_image, finalBoxInfos);
            detector->WaitFinish();
            // manage detection result
#ifdef USE_SDK
            instance_manager->GetInstancesInfo(finalBoxInfos, pose, instance_infos);
#else
            instance_manager->GetInstancesInfo(finalBoxInfos, instance_infos);
#endif

            std::chrono::time_point<std::chrono::system_clock> t2 = std::chrono::system_clock::now();
            float dur = (float)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000;
            std::cout << "duration time:" << dur << "ms" << std::endl;

            // #ifndef USE_SDK
            drawBoxes(finalBoxInfos, raw_image);
            // #else
            // drawInstance(instance_infos, raw_image);
            // #endif
            cv::namedWindow("MNN", CV_WINDOW_NORMAL);
            cv::imshow("MNN", raw_image);
            cv::waitKey(1);
#ifdef __aarch64__
            std::chrono::milliseconds timespan(1000);
            std::this_thread::sleep_for(timespan);
#endif
        }
    }





    return 0;
}
