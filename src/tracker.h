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


class Tracker: public Model{
    public:
        Tracker(std::string& modelName, int width=160, int height=160);
        virtual void PrepareInputAndOutputNames();
        ~Tracker();

        void Track(const cv::Mat& raw_image, std::vector<BoxInfo>& finalBoxInfos);
        virtual void Preprocess(const cv::Mat& raw_image,cv::Mat& image, const BoxInfo& init_state);
        virtual void Postprocess();
        void Init(const cv::Mat& raw_image,const BoxInfo& init_state);
        void CropFromImage(const cv::Mat& raw_image, cv::Mat& image, float crop_size, const std::vector<int>& size);
        virtual void LoadToInputTensors(const cv::Mat& image);
        void GenerateTemplate(const cv::Mat& image);
        void GenerateBoxInfo(std::vector<BoxInfo>& boxInfos, float score_threshold);
    private:
        std::vector<int> mExemplarSize;
        std::vector<int> mInstanceSize;
        std::vector<std::shared_ptr<MNN::Tensor>> mTemplateTensorsHost;
        BoxInfo mState;
        float mContextAmount;
        // model used for generate template
        std::string mTemplateModelName;

        cv::Scalar mChannelAverage;

};
