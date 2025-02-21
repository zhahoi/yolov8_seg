# yolov8_seg
在ros环境下部署yolov8-seg，进行实例分割检测，借助ncnn框架。



## 起

近期的工作都是基于机器人和ROS环境开展的，正在一个熟悉和适应的过程。因为咱的老本行是做视觉的，因此想把ROS环境的视觉检测方面的内容也熟悉一下。本仓库代码的功能是在ROS环境基础下，通过订阅usb_cam的图像节点进行检测推理，最后输出检测结果。

本仓库的检测部分代码基于本人之前上传的仓库：[ncnn_yolov8_seg](https://github.com/zhahoi/ncnn_yolov8_seg)的基础上修改的，如果不想在ros环境下运行，可以参考该仓库的代码实现。



## 承

本仓库代码测试的环境和依赖如下：

- 测试系统版本：Ubuntu 20.04
- ROS系统版本：noetic

本仓库检测部分代码需要依赖opencv和ncnn，以下给定测试版本，需要自行安装，并在`CMakeLists.txt`修改安装路径：

- opencv-3.4.10
- ncnn-20240820-full-source

本仓库图像是从USB摄像头中获取的，因此需要先根据ROS版本安装ROS驱动usb_cam，以下是一个简单的安装参考：

```sh
sudo apt-get install ros-noetic-usb-cam
roscore # 启动ros核心
roslaunch usb_cam usb_cam-test.launch # 另开窗口启动摄像头launch文件，看到自己的大脸表示启动成功
```

参考安装和设置链接：[【Ubuntu】虚拟机安装USB摄像头ROS驱动 usb_cam（最新方法）](https://blog.csdn.net/cnzzs/article/details/142347941)



## 转

以下配置需要根据自己的实际环境修改：

```c++
// 修改到实际的安装路径 (yolov8_seg.h)
#define PARAM_PATH "/home/hit/ncnn_ws/src/yolov8_seg/weights/yolov8s-seg-sim-opt-fp16.param"

#define BIN_PATH "/home/hit/ncnn_ws/src/yolov8_seg/weights/yolov8s-seg-sim-opt-fp16.bin"
```

```cmake
# 根据实际安装的opencv和ncnn路径修改 (CMakeLists.txt)
set(OpenCV_DIR "/home/hit/Softwares/opencv-3.4.10/build")
find_package(OpenCV REQUIRED)

set(ncnn_DIR "/home/hit/Softwares/ncnn/build/install/lib/cmake/ncnn")
find_package(ncnn REQUIRED)
if(NOT TARGET ncnn)
    message(WARNING "ncnn NOT FOUND! Please set ncnn_DIR environment variable")
else()
    message("ncnn found")
endif()
```

本仓库的使用过程如下：

```sh
$ cd ~/catkin_ws/src
$ git clone https://github.com/zhahoi/yolov8_seg.git
...省略配置环境和修改权重路径等操作
$ cd ..
$ catkin_make
$ source devel/setup.bash
$ roslaunch yolov8-seg run.launch
```

启动完launch文件之后，使用以下脚本打开rviz，订阅`Image`，选择`/detected_output`节点便可以看到实际的检测结果了。

```sh
$ rviz
```

![image](https://github.com/zhahoi/yolov8_seg/blob/main/backup/image.png)

## 合

本仓库是本人第一次在ros环境下部署检测任务，老实说不是很难，但是网上有关ros和yolo系列的部署并不多，且大多都是TensorRT之类的推理框架，因此也踩 了很多的坑，不过最后还是成功部署了。

本仓库的代码可以作为参考部署其他的视觉任务，具体的流程大差不差触类旁通，只要搞懂一个大多数就都会了。如果觉得本仓库的代码质量还不错的话，麻烦给一个star或者fork，这是我开源自己代码最大的动力。以上。

