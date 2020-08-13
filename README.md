# DeployedDetector


## Introduce
The Project includes three parts in total.
1. one is for postprocess for detector of `ssd` and `centernet`
2. the other is for instance manager, managing all objects detected from start time
and calculate depth, location, get all visible objects, pass them on to semantic map
3. the last one is for samrpn tracking algorithm.


## Installation

### requirements
1. opencv, build it from source
2. Eigen3
```
sudo apt install libeigen3-dev
```
3. glog, build it from source

4. MNN inference framework
git clone mnn lib from github, compile it and generate
lib and include headers put them in MNN/ directory




### compile
```bash
# build detector
mkdir build && cd build && cmake .. && make -j`nproc`

# if want to use sdk, use instead
cmake .. -DBUILD_SDK=ON

# build tracker
cmake .. -DBUILD_TRACKER=ON
```

## Development
please refer to src/detect.cpp for example,
if you want to change detector type(ssd or centernet)
```cpp
// centernet detector
detector.reset(new CenterNetDetector(model_name));

// ssd detector
detector.reset(new Detector(model_name));
```


