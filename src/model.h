#ifndef __MODEL_H__
#define __MODEL_H__
#include <iostream>
#include <string>
#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"
#include <memory>

class Model{
    public:
        Model(std::string& modelName, int width, int height);

        virtual void PrepareInputAndOutputNames();
        virtual ~Model();
        void SetUpInputAndOutputTensors();

    protected:
        std::string mModelName;
        MNN::ScheduleConfig mConfig;
        // hw
        std::vector<int> mInputSize;
        std::vector<int> mOriginInputSize;


        std::shared_ptr<MNN::Interpreter> mNet;
        MNN::Session* mSession = nullptr;

        // input and output
        std::vector< MNN::Tensor*> mOutputTensors;
        std::vector<MNN::Tensor*> mInputTensors;
        std::vector<MNN::Tensor*> mOutputTensorsHost;

        std::vector<std::string> mInputNames;
        std::vector<std::string> mOutputNames;

};
#endif
