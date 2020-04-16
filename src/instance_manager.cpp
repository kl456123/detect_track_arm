#include "instance_manager.h"
#include "define.h"
#include <assert.h>

using namespace std;



int generateInstanceId(){
    static int instance_id=0;
    std::cout << "------------------------------------------generateInstanceId,id:" << instance_id << std::endl;
    return instance_id++;
}

InstanceManager::InstanceManager(){
    mCameraHeight = 0.15;
    // mIntrinsicMatrix<<489.711, 0,  601.698, 0, 489.605, 443.729 ,0,       0,       1;
    mIntrinsicMatrix = Eigen::Matrix<double, 3, 3>::Identity();
    mTranslationMatrix = Eigen::Matrix<double,3,1>::Zero();
    mRotationMatrix = Eigen::Matrix<double, 3,3>::Identity();
    mDistanceThresh = 2;
    mIou2dThresh = 0.5;
    mUsed2d = false;

    mClipRange = {0.17f, 0.7f};
    mCountThresh = 3;
    // dont work
    mSoftBoundary = {5, 5};
    mMinScale = 0.05;
    mMaxScale = 0.5;
    mDistanceThresh2D = 0.2;

    mAmendThresh = 0.2;
}


#ifdef USE_SDK
void InstanceManager::SetupCamera(const CameraCalibrationParameter& camera_param){
    auto k = camera_param._Kl;
    mIntrinsicMatrix<<k[0], k[1],k[2],k[3],k[4],k[5],k[6],k[7],k[8];
}
void InstanceManager::SetupPose(const ImrPose& pose){
    // assgin
    // T
    mTime  = pose.time;
    auto p = pose._position;
    mTranslationMatrix <<(double)p[0], (double)p[1], (double)p[2];

    // R
    auto r = pose._rotation;
    Eigen::Quaterniond q(r[0], r[1], r[2], r[3]);
    mRotationMatrix = q.matrix();




}
#else
void InstanceManager::SetupCamera(){
    float k[9] = {4.9008025087775349e+02, 0., 6.5899436439430656e+02, 0.,
           4.8978248934587106e+02, 4.1209311026514359e+02, 0., 0., 1.};
    mIntrinsicMatrix<<k[0], k[1],k[2],k[3],k[4],k[5],k[6],k[7],k[8];
}
void InstanceManager::SetupPose(){
    // assgin
    // T
    mTranslationMatrix = Eigen::Matrix<double, 3,1>::Zero();

    // R
    mRotationMatrix = Eigen::Matrix3d::Identity();
}

#endif


void InstanceManager::GetInstancesInfo(const std::vector<BoxInfo>& box_infos,
#ifdef USE_SDK
        ImrPose& pose,
#endif
        std::vector<InstanceInfo>& instance_infos){
    // init matrix param
#ifdef USE_SDK
    SetupPose(pose);
#else
    SetupPose();
#endif
    double s = 2.0;
    std::vector<int> visible_ids;
    GetVisibleInstanceId(visible_ids);
    std::cout<<"Visible ids: "<<visible_ids.size()<<std::endl;
    std::vector<int> updated_ids;
    updated_ids.reserve(visible_ids.size());

    for(auto&box_info: box_infos){
        InstanceInfo instance_info;
        // calculate location in world coords system
        Eigen::MatrixXd points(3, 3);
        Eigen::Vector3d v_3d1((box_info.cx-0.5*box_info.width) * s, s*(box_info.cy  +0.5*box_info.height), 1.0);
        Eigen::Vector3d v_3d2(box_info.cx * s, s*(box_info.cy  +0.5*box_info.height), 1.0);
        Eigen::Vector3d v_3d3((box_info.cx+0.5*box_info.width) * s, s*(box_info.cy  +0.5*box_info.height), 1.0);
        points<<v_3d1, v_3d2, v_3d3;

        {
            // amend it when box near the bounary
            if(box_info.box.x<mSoftBoundary[0]){
                std::cout<<"amend left"<<std::endl;
                //move left
                for(int i=0;i<3;i++){
                    points(0, i) -=box_info.width*mAmendThresh;
                }
            }
            if(box_info.box.y<mSoftBoundary[1]){
                std::cout<<"amend bottom"<<std::endl;
                //move bottom
                for(int i=0;i<3;i++){
                    points(1, i) +=box_info.height*mAmendThresh;
                }
            }

            if(box_info.box.x+box_info.box.width>640-mSoftBoundary[0]){
                std::cout<<"amend right"<<std::endl;
                //move right
                for(int i=0;i<3;i++){
                    points(0, i) +=box_info.width*mAmendThresh;
                }
            }
            // no need to move forward
        }

        // ignore object above the primary point
        float cx = mIntrinsicMatrix(0, 2);
        float cy = mIntrinsicMatrix(1, 2);
        if(v_3d2(1)<cy){
            continue;
        }

        auto tmp = mIntrinsicMatrix.inverse() * points;
        double scale = mCameraHeight/tmp(1);
        auto location = tmp*scale;


        // check depth
        float depth = location(2,1);
        if(depth>mClipRange[1] || depth<mClipRange[0]){
            continue;
        }

        instance_info.depth = depth;

        Eigen::Matrix<double, 3,3> mRotationMatrix22 = Eigen::Matrix<double, 3,3>::Identity();
        mRotationMatrix22<<1,0,0,0,0,1,0,-1,0;

        auto new_location = (mRotationMatrix * mRotationMatrix22 * location).colwise() + mTranslationMatrix;

        instance_info.class_name = box_info.class_name;

        // front face
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                instance_info.front_face_points[i][j] = new_location(j,i);
            }
        }

        float target_scale = (new_location.col(0)-new_location.col(2)).norm();
        instance_info.scale = std::min(std::max(target_scale, mMinScale), mMaxScale);

        // add rear location
        Eigen::Matrix3d rear_location = location;
        for(int i=0;i<3;i++){
            rear_location(2, i) += instance_info.scale;
        }

        rear_location = (mRotationMatrix * mRotationMatrix22 * rear_location).colwise() + mTranslationMatrix;
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                instance_info.rear_face_points[i][j] = rear_location(j,i);
            }
        }

        for(int i=0;i<3;i++){
            instance_info.location[i] = (new_location(i, 1)+rear_location(i,1))/2.0;
        }
        // proj location to image
        {
            Eigen::Vector3d world_point_3d(instance_info.location[0], instance_info.location[1], instance_info.location[2]);
            auto camera_point_3d = (mRotationMatrix*mRotationMatrix22).inverse()*(world_point_3d - mTranslationMatrix);
            auto camera_point_2d_homo = mIntrinsicMatrix*camera_point_3d;
            float depth = camera_point_2d_homo(2);
            float x = camera_point_2d_homo(0)/depth;
            float y = camera_point_2d_homo(1)/depth;
            instance_info.cx = x*0.5;
            instance_info.cy = y*0.5;
        }

        instance_info.box = box_info.box;
        instance_info._count = 0;
        instance_info.valid=false;

        bool matched;
        // finally get the instance id
        instance_info.instance_id = GetInstanceId(instance_info, mUsed2d, matched);
        if(matched){
            updated_ids.push_back(instance_info.instance_id);
        }

        // update by id
        UpdateInstanceInfo(instance_info.instance_id, instance_info);

        instance_infos.push_back(instance_info);
    }

    //updated_ids
    // remove useless instance
    for(int i=0;i<visible_ids.size();i++){
        bool flag = true;
        for(int j=0;j<updated_ids.size();j++){
            if(visible_ids[i]==updated_ids[j]){
                flag = false;
                break;
            }
        }
        if(flag){
            auto& instance = mInstancesMap[visible_ids[i]];
            instance._count--;
            if(instance._count==-mCountThresh){
                mInstancesMap.erase(visible_ids[i]);
            }
        }
    }
    std::cout<<"manager size: "<<mInstancesMap.size()<<std::endl;
    std::cout<<"detect size: "<<instance_infos.size()<<std::endl;
}

void InstanceManager::UpdateInstanceInfo(int instance_id, InstanceInfo& instance_info){
    // update box_2d
    auto& instance = mInstancesMap[instance_id];
    instance.box = instance_info.box;

    // update class count
    if(instance.class_name==instance_info.class_name){
        instance.class_count+=1;
    }else{
        instance.class_count-=1;
    }

    // override class name according to class count(when it equals to zero)
    if(instance.class_count==0){
        instance.class_count=1;
        instance.class_name=instance_info.class_name;
    }


    instance.instance_id = instance_info.instance_id;

    // only update when it is not in the boundary
    // otherwise its scale is not precise due to part visible.
    float x1 = instance_info.box.x;
    float y1 = instance_info.box.y;
    float x2 = x1+instance_info.box.width;
    float y2 = y1+instance_info.box.height;
    bool boundary_cond = x1<=mSoftBoundary[0] || x2>=640-mSoftBoundary[0]||y1<=mSoftBoundary[1]|| y2>=400-mSoftBoundary[1];
    // update scale
    if(!boundary_cond &&instance.depth>instance_info.depth){
        // update location when it can be visible clearly
        instance.depth = instance_info.depth;
        float translate[3];
        for(int i=0;i<3;i++){
            translate[i] = instance_info.location[i] - instance.location[i];
            instance.location[i] = instance_info.location[i];
        }
        // update scale
        instance.scale = instance_info.scale;
        // update front face and rear face
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                instance.front_face_points[i][j] = instance_info.front_face_points[i][j];
            }
        }

        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                instance.rear_face_points[i][j] = instance_info.rear_face_points[i][j];
            }
        }
    }
    instance._count =std::min(instance._count+1, mCountThresh);
    if(instance._count==mCountThresh){
        instance.valid=true;
    }
}

void InstanceManager::GetVisibleInstanceId(std::vector<int>& ids){
    Eigen::Matrix<double, 3,3> mRotationMatrix22 = Eigen::Matrix<double, 3,3>::Identity();
    mRotationMatrix22<<1,0,0,0,0,1,0,-1,0;
    for(auto& iter: mInstancesMap){
        auto& instance_info = iter.second;
        auto location = instance_info.location;


        // world 3d coords to image 2d coords
        Eigen::Vector3d world_point_3d(location[0], location[1], location[2]);
        auto camera_point_3d = (mRotationMatrix*mRotationMatrix22).inverse()*(world_point_3d - mTranslationMatrix);
        auto camera_point_2d_homo = mIntrinsicMatrix*camera_point_3d;
        float depth = camera_point_2d_homo(2);
        float x = camera_point_2d_homo(0)/depth;
        float y = camera_point_2d_homo(1)/depth;
        bool depth_cond = depth<mClipRange[1] && depth>mClipRange[0];

        float x1 = x*0.5-instance_info.box.width*0.5;
        float y1 = y*0.5-instance_info.box.height*0.5;
        float x2 = x*0.5+instance_info.box.width*0.5;
        float y2 = y*0.5+instance_info.box.height*0.5;
        bool boundary_cond = x1<=mSoftBoundary[0] || x2>=640-mSoftBoundary[0]||y1<=mSoftBoundary[1]|| y2>=400-mSoftBoundary[1];
        if(depth_cond && !boundary_cond){
            ids.push_back(instance_info.instance_id);
            instance_info.visible = true;
            instance_info.cx = x*0.5;
            instance_info.cy = y*0.5;
        }else{
            instance_info.visible = false;
        }
    }
}


int InstanceManager::GetInstanceId(const InstanceInfo& target, bool use_2d, bool& matched){
    int instance_id = 0;
    if(use_2d){
        float max_iou = 0;
        // find the nearest instance
        for(auto& iter: mInstancesMap){
            auto& instance_info = iter.second;
            float iou_value = iou(instance_info.box, target.box);
            if(max_iou <iou_value){
                max_iou = iou_value;
                instance_id = instance_info.instance_id;
            }
        }

        // match or insert new instance
        if(max_iou < mIou2dThresh){
            int id = generateInstanceId();
            mInstancesMap.insert(std::make_pair(id,target));
            matched=false;
            return id;
        }else{
            matched=true;
            return instance_id;
        }
    }else{
        float min_dist = 10000;
        // find the nearest instance
        for(auto& iter: mInstancesMap){
            auto& instance_info = iter.second;
            float dist = 0;
            for(int i=0;i<3;i++){
                dist+=std::pow(instance_info.location[i]-target.location[i], 2);
            }
            dist = std::sqrt(dist);
            if(min_dist > dist){
                min_dist = dist;
                instance_id = instance_info.instance_id;
            }
        }

        // match or insert new instance

        matched = false;
        // 3d catch
        if(min_dist < mDistanceThresh* mInstancesMap[instance_id].scale){
            matched = true;
            return instance_id;
        }
        std::cout<<"min_dist: "<<min_dist<<std::endl;
        std::cout<<"scale thresh: "<<mDistanceThresh*mInstancesMap[instance_id].scale<<std::endl;;
        std::cout<<"dist unmatched"<<std::endl;

        // 2d catch
        if(mInstancesMap[instance_id].visible){
            float dist_2d = 0;
            dist_2d+=std::pow((mInstancesMap[instance_id].cx-target.cx), 2);
            dist_2d+=std::pow((mInstancesMap[instance_id].cy-target.cy), 2);

            dist_2d = std::sqrt(dist_2d);
            float size = mDistanceThresh2D*std::sqrt(mInstancesMap[instance_id].box.width*mInstancesMap[instance_id].box.height);
            std::cout<<"size: "<<size<<std::endl;
            std::cout<<"width: "<<mInstancesMap[instance_id].box.width<<std::endl;
            std::cout<<"height: "<<mInstancesMap[instance_id].box.height<<std::endl;
            if(dist_2d<size){
                matched = true;
                std::cout<<"dist_2d matched"<<std::endl;
                return instance_id;
            }
            std::cout<<"dist_2d unmatched"<<std::endl;
        }
        // catch failed, create new
        int id = generateInstanceId();
        mInstancesMap.insert(std::make_pair(id,target));
        matched = false;
        return id;
    }

}



