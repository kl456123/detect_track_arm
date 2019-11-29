#include "tracker.h"
#include <cmath>

Tracker::Tracker(std::string& modelName, int width, int height):Model(modelName, width, height){
    mContextAmount = 0.5;
    mPenaltyK = 0.04;
    mWindowInfluence = 0.4;
    mLR = 0.5;
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
    mOutputNames.push_back("anchors");
    mOutputNames.push_back("window");
}

void Tracker::Postprocess(float scale_z, float image_width, float image_height, BoxInfo& finalBoxInfo){
    std::vector<BoxInfo> boxInfos;
    GenerateBoxInfo(boxInfos, 0);
    float max_pscore = -1;
    int best_idx = 0;
    float best_penalty;
    float best_score;
    // cosine window
    auto window  = mOutputTensorsHost[3]->host<float>();
    for(int i=0;i<boxInfos.size();i++){
        // scale penalty
        float scale_current = get_size(boxInfos[i].width, boxInfos[i].height);
        float scale_last = get_size(mState.box.width*scale_z, mState.box.height*scale_z);
        float scale_cost = (scale_current/scale_last);
        scale_cost = std::max(scale_cost,1/scale_cost);
        // aspect ratio penalty
        float ratio_cost = (float(mState.box.width) / mState.box.height) / (boxInfos[i].width/boxInfos[i].height);
        ratio_cost = std::max(ratio_cost,1/ratio_cost);
        float penalty = std::exp(-(ratio_cost*scale_cost-1)*mPenaltyK);
        float pscore = penalty* boxInfos[i].score;

        // window penalty
        pscore = pscore * (1-mWindowInfluence)+window[i]*mWindowInfluence;
        if(max_pscore<pscore){
            max_pscore = pscore;
            best_idx = i;
            best_penalty = penalty;
            best_score = boxInfos[i].score;
        }
    }

    float w = boxInfos[best_idx].width/scale_z;
    float h = boxInfos[best_idx].height/scale_z;
    float cx = boxInfos[best_idx].cx/scale_z;
    float cy = boxInfos[best_idx].cy/scale_z;
    float lr = best_penalty*best_score*mLR;
    cx+= mState.cx;
    cy+= mState.cy;
    // smooth bbox
    float width = mState.box.width*(1-lr)+w*lr;
    float height = mState.box.height*(1-lr) + h*lr;

    // clip boundary
    cx = std::max(0.f, std::min(cx, image_width));
    cy = std::max(0.f, std::min(cy, image_height));
    width = std::max(10.f, std::min(width, image_width));
    height = std::max(10.f, std::min(height, image_height));

    // update state
    mState.cx = cx;
    mState.cy = cy;
    mState.box.width = width;
    mState.box.height = height;

    finalBoxInfo.cx = cx;
    finalBoxInfo.cy = cy;
    finalBoxInfo.box.width = width;
    finalBoxInfo.box.height = height;
    finalBoxInfo.box.x = cx-width/2;
    finalBoxInfo.box.y = cy-height/2;
    finalBoxInfo.score = best_score;
}

void Tracker::GenerateBoxInfo(std::vector<BoxInfo>& boxInfos, float score_threshold){
    // decode location and scores
    // nx2
    auto scores = mOutputTensorsHost[0]->host<float>();
    auto location = mOutputTensorsHost[1]->host<float>();
    auto anchors = mOutputTensorsHost[2]->host<float>();
    int num = mOutputTensorsHost[0]->batch();
    for(int i=0;i<num;i++){
        // softmax
        auto bg = std::exp(scores[i*2]);
        auto fg = std::exp(scores[i*2+1]);
        fg = fg/(bg+fg);
        BoxInfo box_info;
        box_info.score = fg;
        box_info.cx = location[i*4] * anchors[i*4+2] + anchors[i*4];
        box_info.cy = location[i*4+1] * anchors[i*4+3] + anchors[i*4+1];
        box_info.width = std::exp(location[i*4+2]) * anchors[i*4+2];
        box_info.height = std::exp(location[i*4+3])* anchors[i*4+3];
        boxInfos.push_back(box_info);
    }
}
void Tracker::Track(const cv::Mat& raw_image, std::vector<BoxInfo>& finalBoxInfos){
    float deltas = mContextAmount * (mState.box.width+mState.box.height);
    float w_z = mState.box.width + deltas;
    float h_z = mState.box.height + deltas;
    float s_z = std::sqrt(w_z*h_z);
    float scale_z = mExemplarSize[0]/s_z;
    float s_x = s_z/mExemplarSize[0]* mInstanceSize[0];

    cv::Mat image;
    CropFromImage(raw_image, image, s_x, mInstanceSize);
    Run(image);

    BoxInfo finalBoxInfo;
    Postprocess(scale_z, raw_image.cols, raw_image.rows, finalBoxInfo);
    finalBoxInfos.push_back(finalBoxInfo);
    // update state
}



void Tracker::CropFromImage(const cv::Mat& raw_image, cv::Mat& image, float crop_size, const std::vector<int>& size){
    crop_size = std::round(crop_size);
    float center_x = mState.cx;
    float center_y = mState.cy;
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
    mState.cx = mState.box.x + (float(mState.box.width) - 1 )/2;
    mState.cy = mState.box.y + (float(mState.box.height) - 1 )/2;
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
