#ifndef __CENTERNET_DETECTOR_H__
#define __CENTERNET_DETECTOR_H__
#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include "detector.h"
#include "common.h"

#include <chrono>



class CenterNetDetector: public Detector{
    public:
        CenterNetDetector(std::string& modelName, int width=160, int height=160, float nms_threshold=0.45f, float score_threshold=0.1f);

        virtual void Preprocess(cv::Mat image_in, cv::Mat& image_out);

        void GenerateBoxInfo(std::vector<BoxInfo>& boxInfos, float score_threshold);
        virtual void Detect(const cv::Mat& raw_image, std::vector<BoxInfo>& finalBoxInfos);

        virtual void PrepareInputAndOutputNames()override;
};


#endif
