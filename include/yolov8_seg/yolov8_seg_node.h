#ifndef YOLOV8_SEG_NODE_H
#define YOLOV8_SEG_NODE_H

/// C++ standard headers
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
/// Main NCNN YOLOV8 SEG header
#include "yolov8_seg/yolov8_seg.h"
/// ROS headers
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <cv_bridge/cv_bridge.h>
/// this package headers
#include "yolov8_seg/bbox.h"
#include "yolov8_seg/bboxes.h"


class NcnnYolov8SegRos
{
private:
    ///// YOLO
    int target_size;
    float prob_threshold;
    float nms_threshold;
    bool use_gpu;
    std::string ncnn_param_path;
    std::string ncnn_bin_path;
    std::string image_topic;
    int downsampling_infer;

    std::shared_ptr<Yolo> m_yolo = nullptr;
    int m_counter = 0;

    ///// ros and tf
    ros::NodeHandle m_nh;
    ros::Subscriber m_img_sub;
    ros::Publisher m_detected_img_pub, m_objects_pub;
    ///// Functions
    void ImageCallback(const sensor_msgs::Image::ConstPtr& msg);
    void processImage(cv::Mat& img_in, const double& time);

public:
    NcnnYolov8SegRos(const ros::NodeHandle& n); // constructor
    ~NcnnYolov8SegRos(){}; // destructor
};

#endif