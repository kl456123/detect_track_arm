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

struct BoxInfo {
    cv::Rect box;
    float score;
};



void drawBoxes(std::vector<BoxInfo>& finalBoxInfos, cv::Mat& raw_image);
void *open_video_stream(const char *f, int c, int w, int h, int fps);
void loadOpenCLLib();
float iou(cv::Rect box0, cv::Rect box1);


#endif
