#include <stereo_compute.h>

StereoCompute::StereoCompute(ros::NodeHandle nh): nh_(nh)
{
    image_transport::ImageTransport image_trans(nh_);

}
