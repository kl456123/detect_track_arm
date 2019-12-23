#include <iostream>
#include <memory>
#include "Filter.h"
#include "imrsdk.h"
#include "DoubleBuffer.h"
#include <opencv2/core/core.hpp>


using namespace indem;
class SDK{
    public:
        SDK();
        virtual ~SDK(){};
        static void SdkPoseCallBack(int iType, void* pData, void* pParam);
        static void SdkCameraCallBack(double time, unsigned char* pLeft, unsigned char* pRight, int width, int height, int channel, void* pParam);
        const CameraCalibrationParameter& GetModuleParams();
        const ImrPose& GetPose();
        cv::Mat ReadImage();
        void PrintEach(int row, int col, double* ptr);
        void PrintInfo();

    private:
        std::shared_ptr<CIMRSDK> mpSDK;
        static CameraCalibrationParameter g_params;
        static ImrPose            g_sdk_pos;
        static ImrPose g_sdk_pos_filter;
        static CFilter m_Filter;
        static DoubleBuffer<cv::Mat> g_image_buffer;
};
