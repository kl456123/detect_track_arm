#ifndef __COMMON_H__
#define __COMMON_H__
#include <vector>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <dlfcn.h>
#include "MNNDefine.h"


enum INPUTMODE{
    IMAGE=0,
    VIDEO=1,
};

enum CLASS_NAME{
    BG=0,
    PERSON,
    PET_CAT,
    PET_DOG,
    SOFA,
    TABLE,
    BED,
    EXCREMENT,
    WIRE,
    KEY,
};

struct BoxInfo {
    cv::Rect box;
    float score;
    float cx;
    float cy;
    // in some case, float type is needed
    float width;
    float height;

    // class
    CLASS_NAME class_name;

    // used for record its position
    int index;
};



void drawBoxes(const std::vector<BoxInfo>& finalBoxInfos, cv::Mat& raw_image);

#ifdef USE_SDK
struct InstanceInfo;
void drawInstance(const std::vector<InstanceInfo>&, cv::Mat& raw_image);
#endif
void *open_video_stream(const std::string& f, int c, int w, int h, int fps);
void loadOpenCLLib();
float iou(cv::Rect box0, cv::Rect box1);
float get_size(float w, float h);

void softmax(float* data, int num);


#endif
