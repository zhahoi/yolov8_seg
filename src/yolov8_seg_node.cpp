#include "yolov8_seg/yolov8_seg_node.h"

NcnnYolov8SegRos::NcnnYolov8SegRos(const ros::NodeHandle& n) : m_nh(n)
{
    ///// params
    m_nh.param<int>("/yolov8_seg_ros/target_size", target_size, 320);
    m_nh.param<float>("/yolov8_seg_ros/prob_threshold", prob_threshold, 0.4);
    m_nh.param<float>("/yolov8_seg_ros/nms_threshold", nms_threshold, 0.5);
    m_nh.param<bool>("/yolov8_seg_ros/use_gpu", use_gpu, false);
    m_nh.param<std::string>("/yolov8_seg_ros/image_topic", image_topic, "/usb_cam/image_raw");
    m_nh.param<int>("/yolov8_seg_ros/downsampling_infer", downsampling_infer, 1);

    ///// sub pubs
    m_img_sub = m_nh.subscribe<sensor_msgs::Image>(image_topic, 10, &NcnnYolov8SegRos::ImageCallback, this);
    m_detected_img_pub = m_nh.advertise<sensor_msgs::Image>("/detected_output", 10); 
    
    m_objects_pub = m_nh.advertise<yolov8_seg::bboxes>("/detected_objects", 10);

    ROS_WARN("class heritated, starting node...");
}; // constructor


void NcnnYolov8SegRos::processImage(cv::Mat& img_in, const double& time)
{
    // Note: init yolo only once, if initialized in constructor, it will not work
    if (m_yolo == nullptr)
    {
        m_yolo = std::make_shared<Yolo>(target_size, prob_threshold, nms_threshold, use_gpu);
    }
    // infer and draw
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<Object> objects;
    objects.clear();
    m_yolo->detect(img_in, objects);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    ROS_WARN("Inference duration: %f seconds", elapsed.count());

    // handle output
    yolov8_seg::bboxes out_boxes_;
    out_boxes_.header.stamp = ros::Time().fromSec(time);
    for (size_t i = 0; i < objects.size(); ++i)
    {
        yolov8_seg::bbox out_box_;
        auto object_ = objects[i];
        out_box_.prob = object_.prob;
        out_box_.label = object_.label;
        out_box_.x = object_.rect.x;
        out_box_.y = object_.rect.y;
        out_box_.width = object_.rect.width;
        out_box_.height = object_.rect.height;
        out_box_.class_name = m_yolo->getClassName(object_.label);
        out_boxes_.bboxes.push_back(out_box_);
    }

    // publish
    if (out_boxes_.bboxes.size() > 0)
    {
        m_objects_pub.publish(out_boxes_);
    }

    if (objects.size() > 0) {
        ROS_INFO("Detected %zu objects", objects.size());
    } else {
        ROS_WARN("No objects detected.");
    }

    // draw fps and detect result
    if (objects.size() > 0) {
        m_yolo->draw(img_in, objects);
    }
    else {
        m_yolo->draw_unsupported(img_in);
    }

    m_yolo->draw_fps(img_in);

    // publish image
    cv_bridge::CvImage bridge_img_ = cv_bridge::CvImage(out_boxes_.header, sensor_msgs::image_encodings::BGR8, img_in);
    
    sensor_msgs::Image _raw_img_msg;
    bridge_img_.toImageMsg(_raw_img_msg);
    m_detected_img_pub.publish(_raw_img_msg);

    return;
}

void NcnnYolov8SegRos::ImageCallback(const sensor_msgs::Image::ConstPtr& msg)
{
    ROS_INFO("Received image message");
    m_counter++;
    if (m_counter % downsampling_infer == 0)
    {
        cv::Mat img_in_ = cv_bridge::toCvCopy(*msg, sensor_msgs::image_encodings::BGR8)->image;
        processImage(img_in_, msg->header.stamp.toSec());
    }
    return;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "yolov8_seg");
    ros::NodeHandle n("~");

    NcnnYolov8SegRos ncnn_yolov8_seg_ros(n);

    ros::AsyncSpinner spinner(2); // Use 2 threads
    spinner.start();
    ros::waitForShutdown();

    return 0;
}