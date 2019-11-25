#include "model.h"

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
