#include "tracker.h"
#include <cmath>

Tracker::Tracker(std::string& modelName, int width, int height):Model(modelName, width, height){
    mContextAmount = 0.5;
    mTemplateModelName = "/home/indemind/Models/siamrpn_template.mnn";
    mExemplarSize = {127,127};
    mInstanceSize = {255,255};
    PrepareInputAndOutputNames();
    SetUpInputAndOutputTensors();
}

void Tracker::PrepareInputAndOutputNames(){
    mInputNames = {"input", "zf1", "zf2", "zf3"};
    mOutputNames.push_back("cls");
    mOutputNames.push_back("loc");
}

void Tracker::Postprocess(){
    mOutputTensorsHost[0];

}

void Tracker::GenerateBoxInfo(std::vector<BoxInfo>& boxInfos, float score_threshold){
    // decode location and scores
    // nx2
    auto scores = mOutputTensorsHost[0]->host<float>();
    auto location = mOutputTensorsHost->host<float>();
    auto anchors = mOutputTensorsHost->host<float>();
    int num = mOutputTensorsHost->batch();
    for(int i=0;i<num;i++){
    }
}
void Tracker::Track(const cv::Mat& raw_image, std::vector<BoxInfo>& finalBoxInfos){
    float deltas = mContextAmount * (mState.box.width+mState.box.height);
    float w_z = mState.box.width + deltas;
    float h_z = mState.box.height + deltas;
    float s_z = std::sqrt(w_z*h_z);
    float s_x = s_z/mExemplarSize[0]* mInstanceSize[0];

    cv::Mat image;
    CropFromImage(raw_image, image, s_x, mInstanceSize);
    Run(image);
    Postprocess();
    // update state
}



void Tracker::CropFromImage(const cv::Mat& raw_image, cv::Mat& image, float crop_size, const std::vector<int>& size){
    crop_size = std::round(crop_size);
    float center_x = mState.box.x + (float(mState.box.width) - 1 )/2;
    float center_y = mState.box.y + (float(mState.box.height) - 1 )/2;
    float context_xmin = std::floor(center_x-0.5* crop_size);
    float context_xmax = context_xmin+crop_size-1;
    float context_ymin = std::floor(center_y - 0.5* crop_size);
    float context_ymax = context_ymin+crop_size-1;


    // may be need to pad raw image
    float image_width = raw_image.cols;
    float image_height = raw_image.rows;
    float left_pad = std::max(0.f, -context_xmin);
    float top_pad = std::max(0.f, -context_ymin);
    float right_pad = std::max(0.f, context_xmax-image_width+1);
    float bottom_pad = std::max(0.f, context_ymax-image_height+1);

    cv::Rect roi(context_xmin+left_pad,context_ymin+top_pad, int(crop_size), int(crop_size));

    // bool need_pad = (left_pad>0) or(right_pad>0) or (top_pad>0) or (bottom_pad>0);
    cv::Mat padded_image;
    auto borderType = cv::BORDER_CONSTANT;
    cv::copyMakeBorder(raw_image, padded_image, top_pad, bottom_pad, left_pad, right_pad, borderType, mChannelAverage);

    // crop
    image = padded_image(roi);
    // resize, (w,h)
    cv::resize(image, image, cv::Size(size[0], size[1]));

    // to float
    image.convertTo(image, CV_32FC3);
}

void Tracker::LoadToInputTensors(const cv::Mat& image){
    std::vector<int> dims{1,mInstanceSize[1],mInstanceSize[0], 3};
    auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_data   = nhwc_Tensor->host<float>();
    auto nhwc_size   = nhwc_Tensor->size();
    ::memcpy(nhwc_data, image.data, nhwc_size);
    mInputTensors[0]->copyFromHostTensor(nhwc_Tensor);
    for(int i=0;i<mTemplateTensorsHost.size();i++){
        mInputTensors[i+1]->copyFromHostTensor(mTemplateTensorsHost[i].get());
    }
}

void Tracker::GenerateTemplate(const cv::Mat& image){
    std::vector<int> dims{1,mExemplarSize[1],mExemplarSize[0], 3};
    auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_data   = nhwc_Tensor->host<float>();
    auto nhwc_size   = nhwc_Tensor->size();
    ::memcpy(nhwc_data, image.data, nhwc_size);

    int threads   = 4;
    int precision = 2;

    int forward = MNN_FORWARD_CPU;
    MNN::ScheduleConfig config;
    config.numThread = threads;
    config.type      = static_cast<MNNForwardType>(forward);

    // backend config
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)precision;
    backendConfig.power = MNN::BackendConfig::Power_High;
    config.backendConfig = &backendConfig;

    // create net first
    auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mTemplateModelName.c_str()));

    // create session
    auto sess = net->createSession(config);

    auto inputTensor = net->getSessionInput(sess, NULL);
    inputTensor->copyFromHostTensor(nhwc_Tensor);

    // run
    net->runSession(sess);

    // get output to store
    std::vector<std::string> output_names({"zf1", "zf2", "zf3"});
    for(auto& name: output_names){

        auto t  = net->getSessionOutput(sess, name.c_str());
        auto t_host = std::shared_ptr<MNN::Tensor>(new MNN::Tensor(t, MNN::Tensor::CAFFE));
        t->copyToHostTensor(t_host.get());
        mTemplateTensorsHost.push_back(t_host);
    }
}

void Tracker::Init(const cv::Mat& raw_image, const BoxInfo& info_state){
    cv::Mat image;
    // preprocess
    Preprocess(raw_image, image, info_state);
    GenerateTemplate(image);
}

void Tracker::Preprocess(const cv::Mat& raw_image,cv::Mat& image, const BoxInfo& init_state){
    // get box info from
    mState = init_state;
    // exemplar size in original image
    float deltas = mContextAmount * (mState.box.width+mState.box.height);
    float w_z = mState.box.width + deltas;
    float h_z = mState.box.height + deltas;
    float s_z = std::sqrt(w_z*h_z);
    // mean
    mChannelAverage = cv::mean(raw_image);
    for(int i=0;i<4;i++){
        mChannelAverage[i] = int(mChannelAverage[i]);
    }

    // crop
    CropFromImage(raw_image, image, s_z, mExemplarSize);
}

Tracker::~Tracker(){
}
