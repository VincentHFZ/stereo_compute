#ifndef _STEREO_COMPUTE_H_
#define _STEREO_COMPUTE_H_

// std libs
#include <iostream>
#include <string>
#include <vector>
// ros
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/legacy/compat.hpp>

#define USE_SIFT    1
#define USE_SURF    2
#define USE_ORB     3

class StereoCompute
{
public:
    StereoCompute(ros::NodeHandle nh, int method);
    ~StereoCompute();

public:

private:
    void getCameraParams(void);
    void calibImages(void);
    void getSensorImageCallback(const sensor_msgs::ImageConstPtr& msg);
    void pubCloudPoints(void);

    // get the feature
    void useSift(void);
    void useSurf(void);
    void useOrb(void);

    void matchKeypoints(void);
    int computePose(void);                     // compute stereo pose

private:
    ros::NodeHandle nh_;
    ros::Publisher clouds_pub_;
    image_transport::Subscriber image_sub_;
    cv_bridge::CvImage img_bridge_;

    cv::Size image_size_;
    int n_height;
    int n_width;
    cv::Mat_<double> camParamL;
    cv::Mat_<double> camParamR;
    cv::Mat_<double> distParamL;
    cv::Mat_<double> distParamR;
    cv::Mat_<double> matrixR;
    cv::Mat_<double> matrixT;
    cv::Size imgSize;

    int method_;

    cv::Mat R1;
    cv::Mat R2;
    cv::Mat P1;
    cv::Mat P2;
    cv::Mat Q;

    cv::Mat mapLx, mapLy, mapRx, mapRy;

    cv::Mat image_;
    cv::Mat image_left_;
    cv::Mat image_right_;

    std::vector<cv::KeyPoint> key_points_l, key_points_r;
    cv::Mat descriptors_l, descriptors_r;
    std::vector<cv::DMatch> matches;
    std::vector<cv::Point2f> src_l, src_r;
    std::vector<cv::Vec3f> _3d_points;
    // std::vector<cv::Point2f> dst_l, dst_r;

};




#endif
