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

class StereoCompute
{
public:
    StereoCompute(ros::NodeHandle nh);
    ~StereoCompute();

public:

private:
    ros::NodeHandle nh_;

};




#endif
