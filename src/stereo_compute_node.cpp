#include <iostream>
#include <vector>
#include <string>
// ros
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>             // 点云
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

#include <stereo_compute/stereo_compute.h>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "stereo_compute");
    ros::NodeHandle nh;

    std::string param;
    int method;

    if(argc > 1)
    {
        param = argv[1];
        if(param == "use_sift")
            method = USE_SIFT;
        else if(param == "use_surf")
            method = USE_SURF;
        else if(param == "use_orb")
            method = USE_ORB;
        else
            method = USE_SURF;
    }
    else
        method = USE_SURF;

    // std::cout << "method: " << method << " " << param << " " << argc << std::endl;

    StereoCompute stereo_compute(nh, method);

    while(ros::ok())
    {
        ros::spinOnce();
    }
}
