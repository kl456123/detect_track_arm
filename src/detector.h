#ifndef __DETECTOR_H_
#define __DETECTOR_H_
#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include "model.h"
#include "common.h"

#include <chrono>



class Detector: public Model{
    public:
        Detector(std::string& modelName, int width=160, int height=160, float nms_threshold=0.45f, float score_threshold=0.3f);

        virtual void Preprocess(cv::Mat image_in, cv::Mat& image_out);

        virtual void LoadToInputTensors(cv::Mat& image);

        virtual void LoadToOutputTensors();

        void GetTopK(std::vector<BoxInfo>& input, std::vector<BoxInfo>& output, int top_k);
        void NMS(std::vector<BoxInfo>& boxInfos,std::vector<BoxInfo>& boxInfos_left, float threshold);

        void GenerateBoxInfo(std::vector<BoxInfo>& boxInfos, float score_threshold);
        virtual void Detect(const cv::Mat& raw_image, std::vector<BoxInfo>& finalBoxInfos);

        virtual void PrepareInputAndOutputNames()override;


    private:
        std::vector<float> mVariance;

        int mTopK = 100;
        float mScoreThreshold;
        float mNMSThreshold;

};


#endif
