#ifndef _IMRSDK_IMPL_H_
#define _IMRSDK_IMPL_H_
#include <vector>
#include <thread>
#include "imrdata.h"
#include "config.h"
#include "imrsdk.h"
#include <unordered_map>
#include <future>
#include "Dispather.h"
#include "structdata.h"
#include <Dll.h>
#include "ThreadPool.h"
#include "SlamPlugin.h"
#include "Filter.h"

typedef void(*DataCallback)(int, void*,void*);
typedef void(*ImgCallback)(double, ImrImage, void*);
namespace g3 {
	class LogWorker;
}
struct CPack;

enum CallbackSockEnum {
    CAMERA_CALLBACK_SOCK = 0,     //相机
    HEAD_CALLBACK_SOCK,         //头显IMU
    HOTPLUG_CALLBACK_SOCK,      //热插拔
    HAND_CALLBACK_SOCK,         //手柄
    CALLBACK_SOCK_LENGTH
};

namespace indem {
    struct IMUData;
    class IAlgorithmPlugin;
    class ISlamPlugin;
    class IDriverInterface;
    class IDispather;
    class CVideoCapture;
    class CVideoPlay;
    typedef std::tuple<SLAM_PURE_OUT, HEADSET_OUT, int,IMUData> LoopCache;

    void EmptyImageCallback(double time, unsigned char* pLeft, unsigned char* pRight, int width, int height, int channel, void* pParam);
    void EmptyIMUCallback(double time, float accX, float accY, float accZ, float gyrX, float gyrY, float gyrZ, void* pParam);

	class CIMRSDKImpl {
	public:
		
		static int s_iCBSock[CALLBACK_SOCK_LENGTH];
		CIMRSDKImpl();
		~CIMRSDKImpl();

        //bool CheckLicense(const char* lisence);

		//初始化配置，并启动数据接收线程及算法线程
		bool Init(MRCONFIG config);
		/*
		 * \brief 添加数据回调函数
		 * \param deviceID 设备ID编号
		 * \param cb 对应算法处理完毕后，用于数据传出的回调函数
		 * \param pData 传给回调函数使用的参数
		 */
		void AddDataCallback(int deviceID,DataCallback cb,void* pData);

        void SetOriginalIMUCallback(ModuleIMUCallback cb, void* pData);
        void SetOriginalIMGCallback(ModuleImageCallback cb, void* pData);

		static void RegistDisconnectCallback(HotplugCallback cb, void* param);

        CameraCalibrationParam GetModuleParams();
        ModuleParameters GetModuleDetailInfo();

        void SetHMDDeviceStatus(const char* deviceID, bool master);
        int GetHMDDeviceStatus(const char* deviceID);

        void BeforeSlamInit(void* pParam);
        bool ResetCenter();
        bool ReInitSlam(double t, float* pos, float* quart, float* speed, float* bias);
        bool ImportMap(const char* fullpath);
        bool ExportMap(const char* fullpath);

        int AddPluginCallback(const char* pluginName, const char* callbackName, PluginCallback cb, void* param);
        int InvokePluginMethod(const char* pluginName, const char* methodName, void* inParam, void* outParam);
        static void ListPluginsInfo(int& pluginNum, char** pluginsName);
        static void ListPluginInfo(const char* pluginsName, int& major, int& minor, char* developer);
        void Release();
		
	private:
		//算法线程,动态执行相应的算法
		void ThreadAlgorithm(int );
        //创建数据转发
        void CreateDataDispatch();

		void AlgorithmInvoke(const CPackInternal&, LoopCache& tResult);
        ImrPose GetPoseCallback(double current, ISlamPlugin* pSlam);

		void InitCallbackSock(int i);
		void ReleaseCallbackSock(int i);

        bool InitDriver();
        IDriverInterface* LoadDriver();
        void ReleaseDriver();
        //载入补偿参数
        void InitParams(const char* fullpath);

        //导出配置文件
        //bool ExportConfig(const char* root);
        bool GenerateSlamYaml(const std::string& fullpath);

		//readfile
		void ReadFileThread();

        void InitAlgorithm(IDispather* alg);

        static void SetPluginSearchOrder(const std::string&);
        bool LoadSlamPlugin(const std::string fullpath);
        void ReleaseSlamPlugin();
        //加载算法插件
        void LoadPlugins();
        //卸载算法插件
        void ReleasePlugins();
        //根据配置初始化数据捕获对象
        void CreateCaptureObject(MRCONFIG config = { 0 });
        
        void OriginalIMUCallback(const IMUData& hmdIMU);
        void OriginalImageCallback(const IMUData&);
        void InitModuleParams();
	private:
		bool m_bExit;
		std::unique_ptr<g3::LogWorker> m_pLogger;			//日志
		static std::pair<HotplugCallback, void*> m_hotplug;
		
	private:
        int m_iPipe[2];             //管道
        int m_iSwitch;              //算法开关sock
        bool m_bSDKSwitch = false;  //SDK开关
        bool m_iMaster;             //是否设置了本机主控
        CVideoCapture* m_pCapture;  //数据捕获
        CVideoPlay* m_pPlayer;      //数据播放

        std::thread* m_threadMain = NULL;   //数据分发线程
        std::thread m_tReplay;      //回放线程
        typedef Dll SharedLibrary;
        typedef std::unordered_map<std::string, std::tuple<SharedLibrary, IAlgorithmPlugin*> > AlgorithmPlugins;
        std::atomic<bool> m_bPluginLoad;    //新插件载入中
        AlgorithmPlugins m_mPlugins;    //第三方插件算法
        bool m_bUseSlam;                //是否使用Slam
        std::pair<SharedLibrary, ISlamPlugin*> m_pSlamPlugin;  //Slam插件
        void* m_pBeforeSlamInitParams=nullptr;              //slam初始化之前的参数
        std::pair<DataCallback,void*> m_pSlamCallback;       //slam结果回调函数
        ThreadPool m_tPool;
		//驱动相关
        std::pair<SharedLibrary, IDriverInterface*> m_pDriver;
		
        typedef void(CIMRSDKImpl::*InternalDataCallback)(const IMUData&);
        std::unordered_map<int32_t, InternalDataCallback> m_mCallbackExecutor;
        std::pair<ModuleIMUCallback, void*>        m_pIMUCallback;      //IMU
        std::pair<ModuleImageCallback, void*>      m_pIMGCallback;      //图像
        std::pair<ModuleIMUCallback, void*>        m_pIMUCallbackAsync; //IMU异步
        std::pair<ModuleImageCallback, void*>      m_pIMGCallbackAsync; //图像异步
        CFilter m_fPoseFrequency;           //获取位姿的时间间隔:ms

		double last_imu_time = 0.0; //上一次imu信息
        CameraCalibrationParam m_mCCP;  //相机标定参数
	};

}
#endif