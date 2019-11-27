#include "model.h"
#define PRINT_INFERNCE_TIME

Model::Model(std::string& modelName, int width, int height){

    mModelName = modelName;
    int threads   = 4;
    int precision = 2;

    int forward = MNN_FORWARD_CPU;
    mConfig.numThread = threads;
    mConfig.type      = static_cast<MNNForwardType>(forward);

    // backend config
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)precision;
    backendConfig.power = MNN::BackendConfig::Power_High;
    mConfig.backendConfig = &backendConfig;

    // input config
    mInputSize.push_back(height);
    mInputSize.push_back(width);

    // PrepareInputAndOutputNames();

    // create net first
    mNet = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(modelName.c_str()));

    // create session
    mSession = mNet->createSession(mConfig);
}

void Model::SetUpInputAndOutputTensors(){
    // get input tensor
    if(mInputNames.size()==0){
        mInputTensors.push_back(mNet->getSessionInput(mSession, NULL));
    }else{
        for(auto& input_name: mInputNames){
            mInputTensors.push_back(mNet->getSessionInput(mSession, input_name.c_str()));
        }
    }

    // get output tensor
    for(auto& output_name: mOutputNames){
        auto t = mNet->getSessionOutput(mSession, output_name.c_str());
        mOutputTensors.push_back(t);
        mOutputTensorsHost.push_back(new MNN::Tensor(t, t->getDimensionType()));
    }
}

void Model::PrepareInputAndOutputNames(){
    // mOutputNames.push_back("cls_logits");
    // mOutputNames.push_back("bbox_preds");
    // mOutputNames.push_back("anchors");
}

Model::~Model(){
    mNet->releaseModel();
    mNet->releaseSession(mSession);
    for(auto&t: mOutputTensorsHost){
        delete t;
    }
}

void Model::LoadToInputTensors(const cv::Mat& image){
#ifdef PRINT_INFERNCE_TIME
    auto t1 = std::chrono::system_clock::now();
    // auto start = clock();
#endif
    // copy to tensor
    std::vector<int> dims{1,mInputSize[0] , mInputSize[1], 3};
    auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_data   = nhwc_Tensor->host<float>();
    auto nhwc_size   = nhwc_Tensor->size();
    ::memcpy(nhwc_data, image.data, nhwc_size);

    mInputTensors[0]->copyFromHostTensor(nhwc_Tensor);

#ifdef PRINT_INFERNCE_TIME
    // clock_t end = clock();
    // float duration = float(end - start)/CLOCKS_PER_SEC * 1000;
    // std::cout<<"infer time: "<<duration<<" ms"<<std::endl;
    std::chrono::time_point<std::chrono::system_clock> t2 = std::chrono::system_clock::now();
    float dur = (float)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000;
    std::cout << "input time:" << dur << "ms" << std::endl;
#endif
}

void Model::LoadToOutputTensors(){
#ifdef PRINT_INFERNCE_TIME
    auto t1 = std::chrono::system_clock::now();
#endif
    for(int i=0;i<mOutputTensors.size();i++){
        auto t = mOutputTensors[i];
        auto t_host = mOutputTensorsHost[i];
        t->copyToHostTensor(t_host);
    }
#ifdef PRINT_INFERNCE_TIME
    std::chrono::time_point<std::chrono::system_clock> t2 = std::chrono::system_clock::now();
    float dur = (float)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000;
    std::cout << "output time:" << dur << "ms" << std::endl;
#endif
}

void Model::Run(const cv::Mat& image){
    LoadToInputTensors(image);

#ifdef PRINT_INFERNCE_TIME
    auto t1 = std::chrono::system_clock::now();
    // auto start = clock();
#endif
    // run session
    mNet->runSession(mSession);
#ifdef PRINT_INFERNCE_TIME
    std::chrono::time_point<std::chrono::system_clock> t2 = std::chrono::system_clock::now();
    float dur = (float)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000;
    std::cout << "infer time:" << dur << "ms" << std::endl;
#endif

    LoadToOutputTensors();
}
