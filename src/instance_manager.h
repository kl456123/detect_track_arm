#ifndef __INSTANCE_MANAGER_H__
#define __INSTANCE_MANAGER_H__
#include "common.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <map>
#include <Eigen/Core>
#include <Eigen/Dense>

#ifdef USE_SDK
#include "imrdata.h"
#endif

struct InstanceInfo{
    // class
    CLASS_NAME class_name;

    // instance
    int instance_id;

    // xyz
    float location[3];

    // time
    float time;

    // scale
    float scale;

    // used calc iou in image
    cv::Rect box;

    int _count;
    float front_face_points[3][3];
    float rear_face_points[3][3];
    bool visible;

    int class_count=0;

    bool valid;

    // calc from location instead of center of box
    float cx;
    float cy;

    // only used for updating location
    float depth;
};


class InstanceManager{

    public:
        InstanceManager();
        virtual ~InstanceManager(){};

        int GetInstanceId(const InstanceInfo& target, bool, bool&);
#ifdef USE_SDK
        void SetupPose(const ImrPose& );
        void SetupCamera(const CameraCalibrationParameter&);
void GetInstancesInfo(const std::vector<BoxInfo>& box_infos,ImrPose& pose, std::vector<InstanceInfo>& instance_infos);
#else
        void SetupPose();
        void SetupCamera();
void GetInstancesInfo(const std::vector<BoxInfo>& box_infos,std::vector<InstanceInfo>& instance_infos);
#endif

        void UpdateInstanceInfo(int instance_id, InstanceInfo& instance_info);
        void GetVisibleInstanceId(std::vector<int>& ids);

        void GetInstanceList(std::vector<InstanceInfo>& instances_info){
            instances_info.clear();
            for(auto& iter: mInstancesMap){
                std::cout<<"instance count: "<<iter.second._count<<std::endl;
                if(iter.second.valid){
                    instances_info.push_back(iter.second);
                }
            }
        }

    private:
        std::map<int, InstanceInfo> mInstancesMap;
        float mCameraHeight;
        Eigen::Matrix<double, 3, 3> mIntrinsicMatrix;
        Eigen::Matrix<double, 3, 1> mTranslationMatrix;
        Eigen::Matrix<double, 3, 3> mRotationMatrix;
        float mDistanceThresh;
        float mIou2dThresh;
        bool mUsed2d;
        float mTime;
        float mMinScale;
        float mMaxScale;
        bool mIgnoreFlyThing;
        std::vector<float> mClipRange;
        int mCountThresh;
        std::vector<float> mSoftBoundary;
        float mDistanceThresh2D;
        float mAmendThresh;
};
#endif
