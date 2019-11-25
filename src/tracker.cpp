#include "tracker.h"

Tracker::Tracker(std::string& modelName, int width, int height):Model(modelName, width, height){
    mModelName = modelName;
    PrepareInputAndOutputNames();
    SetUpInputAndOutputTensors();
}

void Tracker::PrepareInputAndOutputNames(){
    mInputNames.push_back("input");
    mInputNames.push_back("zf1");
    mInputNames.push_back("zf2");

    mOutputNames.push_back("cls");
    mOutputNames.push_back("loc");
}


void Tracker::Track(const cv::Mat& raw_image, std::vector<BoxInfo>& finalBoxInfos){

}

void Tracker::Preprocess(const cv::Mat& raw_image, cv::Mat& image){
}

Tracker::~Tracker(){
}
