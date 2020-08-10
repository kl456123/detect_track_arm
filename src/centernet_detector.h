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
#include "opencl/gpu_types.h"

#include <chrono>

using namespace opencl;

class CenterNetDetector: public Detector{
    public:
        CenterNetDetector(std::string& modelName, int width=160, int height=160, float nms_threshold=0.45f, float score_threshold=0.3f);
        virtual ~CenterNetDetector();

        virtual void Preprocess(cv::Mat image_in, cv::Mat& image_out);

        void GenerateBoxInfo(std::vector<BoxInfo>& boxInfos, float score_threshold);
        virtual void Detect(const cv::Mat& raw_image, std::vector<BoxInfo>& finalBoxInfos);

        virtual void PrepareInputAndOutputNames()override;
    private:

        DeviceContext* device_context_;

        bool* res_maxpool_=nullptr;
};


#endif
