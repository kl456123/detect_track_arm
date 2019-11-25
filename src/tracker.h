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
        void Preprocess(const cv::Mat& raw_image, cv::Mat& image);
    private:
        std::string mModelName;

};
