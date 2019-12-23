#include "sdk.h"

CameraCalibrationParameter SDK::g_params;
ImrPose SDK::g_sdk_pos;
ImrPose SDK::g_sdk_pos_filter;
CFilter SDK::m_Filter(80*1);
DoubleBuffer<cv::Mat> SDK::g_image_buffer;

void SDK::SdkPoseCallBack(int iType, void* pData, void* pParam)
{
    ImrModulePose* pos = (ImrModulePose*)pData;
    g_sdk_pos = pos->_pose;
}

void SDK::SdkCameraCallBack(double time, unsigned char* pLeft, unsigned char* pRight, int width, int height, int channel, void* pParam)
{
    if (!m_Filter.IsPass(time)) return;

    g_sdk_pos_filter = g_sdk_pos;
    cv::Mat frame = cv::Mat(height,width,CV_8U,pLeft);
    //resize(frame,frame,cv::Size(320,200));
    g_image_buffer.Write(frame);
    //std::cout << "----------------- IMG Come: " << time<<" -------------------------------"<< std::endl;
}

void SDK::PrintEach(int row, int col, double* ptr)
{
    for (int r = 0; r < row; ++r)
    {
        for (int c = 0; c < col; ++c)
        {
            std::cout << ptr[r*col + c] << "\t";
        }
        std::cout << std::endl;
    }
}

void SDK::PrintInfo()
{
    std::cout << "ACC: " << std::endl;
    PrintEach(3, 4, g_params._Acc);
    std::cout << "Gyr: " << std::endl;
    PrintEach(3, 4, g_params._Gyr);
    std::cout << "Dl: " << std::endl;
    PrintEach(4, 1, g_params._Dl);
    std::cout << "Dr: " << std::endl;
    PrintEach(4, 1, g_params._Dr);
    std::cout << "Kl: " << std::endl;
    PrintEach(3, 3, g_params._Kl);
    std::cout << "Kr: " << std::endl;
    PrintEach(3, 3, g_params._Kr);
    std::cout << "Pl: " << std::endl;
    PrintEach(3, 4, g_params._Pl);
    std::cout << "Pr: " << std::endl;
    PrintEach(3, 4, g_params._Pr);
    std::cout << "Rl: " << std::endl;
    PrintEach(3, 3, g_params._Rl);
    std::cout << "Rr: " << std::endl;
    PrintEach(3, 3, g_params._Rr);
    std::cout << "TSCl: " << std::endl;
    PrintEach(4, 4, g_params._TSCl);
    std::cout << "TSCr: " << std::endl;
    PrintEach(4, 4, g_params._TSCr);
    std::cout << "Baseline: " << g_params._baseline << " m" << std::endl;
    std::cout << "AMax: " << g_params._AMax << std::endl;
    std::cout << "SigmaAC: " << g_params._SigmaAC << std::endl;
    std::cout << "SigmaBa: " << g_params._SigmaBa << std::endl;
    std::cout << "GMax: " << g_params._GMax << std::endl;
    std::cout << "SigmaAwC: " << g_params._SigmaAwC << std::endl;
    std::cout << "SigmaBg: " << g_params._SigmaBg << std::endl;
    std::cout << "SigmaGC: " << g_params._SigmaGC << std::endl;
    std::cout << "SigmaGwC: " << g_params._SigmaGwC << std::endl;
}



SDK::SDK(){
    mpSDK.reset(new CIMRSDK());
    MRCONFIG config = {0};
    config.bSlam = false;
    config.imgResolution = IMG_640;
    //config.imuFrequency = 2;
    //config.imgFrequency = 40;
    mpSDK->Init(config);
    mpSDK->RegistModulePoseCallback(SdkPoseCallBack,NULL);
    mpSDK->RegistModuleCameraCallback(SdkCameraCallBack,NULL);
    g_params = mpSDK->GetModuleParams();
}

const CameraCalibrationParameter& SDK::GetModuleParams(){
    return g_params;
}

const ImrPose& SDK::GetPose(){
    return g_sdk_pos;
}

cv::Mat SDK::ReadImage(){
while(true){
    cv::Mat mat;
    mat = g_image_buffer.Read();
    if(mat.rows>0){
       return mat;
   }
}
}
