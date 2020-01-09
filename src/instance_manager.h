#ifndef __INSTANCE_MANAGER_H__
#define __INSTANCE_MANAGER_H__
#include "common.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <map>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "imrdata.h"

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
    bool visible;
};


class InstanceManager{

    public:
        InstanceManager();
        virtual ~InstanceManager(){};

        int GetInstanceId(const InstanceInfo& target, bool, bool&);
        void SetupPose(const ImrPose& );
        void GetInstancesInfo(const std::vector<BoxInfo>& box_infos,ImrPose& pose, std::vector<InstanceInfo>& instance_infos);
        void UpdateInstanceInfo(int instance_id, InstanceInfo& instance_info);
        void SetupCamera(const CameraCalibrationParameter&);
        void GetVisibleInstanceId(std::vector<int>& ids);

        void GetInstanceList(std::vector<InstanceInfo>& instances_info){
            instances_info.clear();
            for(auto& iter: mInstancesMap){
                if(iter.second.visible){
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
        int mInitCount;
};
#endif
