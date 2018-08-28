#include <stereo_compute/stereo_compute.h>

StereoCompute::StereoCompute(ros::NodeHandle nh, int method): nh_(nh), method_(method)
{
    // initilize image
    n_height = 640;
    n_width = 480;
    image_size_ = cv::Size(n_height, n_width);
    image_left_.create(imgSize, CV_8UC1);
    image_right_.create(imgSize, CV_8UC1);
    image_.create(n_height, n_width * 2, CV_8UC1);

    image_transport::ImageTransport image_trans(nh_);
    image_sub_ = image_trans.subscribe("stereo_raw_image", 1, &StereoCompute::getSensorImageCallback, this);
    img_bridge_ = cv_bridge::CvImage(std_msgs::Header(), "8UC1", image_);
    clouds_pub_ = nh_.advertise<sensor_msgs::PointCloud>("clouds", 50);

    getCameraParams();
    calibImages();
}

StereoCompute::~StereoCompute()
{
}

// get raw image and compute sterep pose
void StereoCompute::getSensorImageCallback(const sensor_msgs::ImageConstPtr& image_msg)
{
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::TYPE_8UC1);
    }
    catch(cv_bridge::Exception& ex)
    {
        ROS_ERROR("cv_bridge exception error: %s", ex.what());
        return;
    }
    cv_ptr->image.copyTo(image_);

    cv::Mat roi_l = image_(cv::Rect(0, 0, 640, 480));       // x y width height
    cv::Mat roi_r = image_(cv::Rect(640, 0, 640, 480));

    roi_l.copyTo(image_left_);
    roi_r.copyTo(image_right_);

    computePose();                                          // 计算位姿
    pubCloudPoints();                                       // 发布点云数据

    // debug
//    cv::imshow("image_left", image_left_);
//    cv::imshow("image_right", image_right_);
//    cv::waitKey(1);
}

// 加载相机参数
void StereoCompute::getCameraParams(void)
{
    camParamL.create(3, 3);
    camParamL << 465.5507, -0.0875, 369.1928,
            0.0, 464.4618, 269.8151,
            0.0, 0.0, 1.0;
    camParamR.create(3, 3);
    camParamR << 465.1805, -0.2614, 386.3437,
            0.0, 464.0834, 236.4345,
            0.0, 0.0, 1.0;
    distParamL.create(5, 1);
    distParamL << -0.4203, 0.2083, 8.3399e-4, -0.0018, -0.0562;
    distParamR.create(1, 5);
    distParamR << -0.4172, 0.1969, 0.0013, -0.0013, -0.0478;
    matrixR.create(3, 3);
    matrixR << 1.0000, 0.0061, -0.0016,
            -0.0061, 0.9999, -0.0119,
            0.0015, 0.0120, 0.9999;
    matrixT.create(3, 1);
    matrixT << -98.8830, -0.3284, -0.6657;

    // debug
    std::cout << "camParamL: \n" << camParamL << std::endl;
    std::cout << "camParamR: \n" << camParamR << std::endl;
    std::cout << "disPraramL: \n" << distParamL << std::endl;
    std::cout << "disPraramR: \n" << distParamR << std::endl;
    std::cout << "matrixR: \n" << matrixR << std::endl;
    std::cout << "matirxT: \n" << matrixT << std::endl;
}

void StereoCompute::calibImages(void)
{
    cv::Rect validROIL, validROIR;

    cv::stereoRectify(camParamL, distParamL,
                      camParamR, distParamR,
                      image_size_, matrixR, matrixT, R1, R2, P1, P2, Q,
                      cv::CALIB_ZERO_DISPARITY, 0, image_size_, &validROIL, &validROIR);
    cv::initUndistortRectifyMap(camParamL, distParamL, R1, P1, image_size_, CV_32FC1, mapLx, mapLy);
    cv::initUndistortRectifyMap(camParamR, distParamR, R2, P2, image_size_, CV_32FC1, mapRx, mapRy);
}

// 使用 sift 提取特征点和特征描述符
void StereoCompute::useSift(void)
{
    key_points_l.clear();
    key_points_r.clear();

    cv::SiftFeatureDetector detector(800);
    detector.detect(image_left_, key_points_l);
    detector.detect(image_right_, key_points_r);

    cv::SiftDescriptorExtractor extractor;
    extractor.compute(image_left_, key_points_l, descriptors_l);
    extractor.compute(image_right_, key_points_r, descriptors_r);
}

// 使用 surf 提取特征点和特征描述符
void StereoCompute::useSurf(void)
{
    key_points_l.clear();
    key_points_r.clear();

    cv::SurfFeatureDetector detector(800);
    detector.detect(image_left_, key_points_l);
    detector.detect(image_right_, key_points_r);

    cv::SurfDescriptorExtractor extractor;
    extractor.compute(image_left_, key_points_l, descriptors_l);
    extractor.compute(image_right_, key_points_r, descriptors_r);
}

// 使用 orb 提取特征点和特征描述符
void StereoCompute::useOrb(void)
{
    key_points_l.clear();
    key_points_r.clear();

    cv::ORB orb;
    orb.detect(image_left_, key_points_l);
    orb.detect(image_right_, key_points_r);
    orb.compute(image_left_, key_points_l, descriptors_l);
    orb.compute(image_right_, key_points_r, descriptors_r);
}

// 特征点匹配
void StereoCompute::matchKeypoints(void)
{
    matches.clear();
    points_l.clear();
    points_r.clear();
    // 匹配特征点，主要计算两个特征点特征向量的欧式距离，当距离小于某个阈值时则认为无匹配
    cv::BruteForceMatcher<cv::L2<float> > matcher;
    matcher.match(descriptors_l, descriptors_r, matches);

    // RANSAC
    // step 1 根据 matches 将特征点对齐，将坐标转换为 float 类型
    std::vector<cv::KeyPoint> R_keypoints_l, R_keypoints_r;
    for(size_t i = 0; i < matches.size(); i++)
    {
        R_keypoints_l.push_back(key_points_l[matches[i].queryIdx]);
        R_keypoints_r.push_back(key_points_r[matches[i].trainIdx]);
    }
    // 坐标转换
    std::vector<cv::Point2f> p01, p02;
    for(size_t i = 0; i <  matches.size(); i++)
    {
        p01.push_back(R_keypoints_l[i].pt);
        p02.push_back(R_keypoints_r[i].pt);
    }
    // 剔除误匹配点
    std::vector<uchar> RansacStatus;
    // cv::Mat Fundamental = cv::findFundamentalMat(p01, p02, RansacStatus, cv::FM_RANSAC);    // 基础矩阵
    cv::Mat Homography = cv::findHomography(p01, p02, RansacStatus, cv::FM_RANSAC); // 单应矩阵
    // 重新定义 RR_keypoint 和 RR_matches 来存储新的关键点和匹配矩阵
    std::vector<cv::KeyPoint> RR_keypoints_l, RR_keypoints_r;
    std::vector<cv::DMatch> RR_matches;
    int iidex = 0;
    for(size_t i = 0; i < matches.size(); i++)
    {
        if(RansacStatus[i] != 0)
        {
            RR_keypoints_l.push_back(R_keypoints_l[i]);
            RR_keypoints_r.push_back(R_keypoints_r[i]);
            matches[i].queryIdx = iidex;
            matches[i].trainIdx = iidex;
            RR_matches.push_back(matches[i]);
            iidex++;
        }
    }

    // debug
//    cv::Mat image_RR_matches;
//    cv::drawMatches(image_left_, RR_keypoints_l,
//                    image_right_, RR_keypoints_r,
//                    RR_matches, image_RR_matches);
//    cv::imshow("result", image_RR_matches);

    for(size_t i = 0; i < RR_keypoints_l.size(); i++)
    {
        points_l.push_back(RR_keypoints_l[i].pt);
        points_r.push_back(RR_keypoints_r[i].pt);
    }

}

// 计算位姿
int StereoCompute::computePose(void)
{
    // 判断是否有图像
    if(image_left_.empty())
    {
        ROS_WARN("No left camera's image");
        return -1;
    }
    if(image_right_.empty())
    {
        ROS_WARN("No right camera's image");
        return -1;
    }

    // calib image
    cv::remap(image_left_.clone(), image_left_, mapLx, mapLy, cv::INTER_LINEAR);
    cv::remap(image_right_.clone(), image_right_, mapRx, mapRy, cv::INTER_LINEAR);

    // 提取特征点
    if(method_ == USE_SIFT)
        useSift();
    else if(method_ == USE_SURF)
        useSurf();
    else if(method_ == USE_ORB)
        useOrb();
    else
    {
        ROS_ERROR("Invalid method");
        return -2;
    }

    // 匹配特征点
    matchKeypoints();

    // 立体结算
    cv::Matx44f _Q;
    Q.convertTo(_Q, CV_64F);

    cv::Vec3f _3d_point;
    _3d_points.clear();
    for(size_t i = 0; i < points_l.size(); i++)
    {
        double d = points_l[i].x - points_r[i].x;
        cv::Vec4f homg_pt = _Q * cv::Vec4f(points_l[i].x, points_l[i].y, d, 1);
        _3d_point = cv::Vec3f(homg_pt.val);
        _3d_point /= homg_pt[3];

        _3d_points.push_back(_3d_point);

        // debug s
//        std::cout << "_3dPoints: \t"
//                  << "x: " << _3d_point[0] << "\t"
//                  << "y: " << _3d_point[1] << "\t"
//                  << "z: " << _3d_point[2] << "\t"
//                  << std::endl;
    }

    return 0;
}

// 发布点云数据
void StereoCompute::pubCloudPoints(void)
{
    sensor_msgs::PointCloud point_clouds;
    point_clouds.header.frame_id = "base_link";//"camera_link";
    point_clouds.header.stamp = ros::Time::now();
    point_clouds.points.resize(_3d_points.size());

    point_clouds.channels.resize(1);
    point_clouds.channels[0].name = "intensity";
    point_clouds.channels[0].values.resize(_3d_points.size());

    for(size_t i = 0; i < _3d_points.size(); i++)
    {
        point_clouds.points[i].x = _3d_points[i][0] / 1000.0;
        point_clouds.points[i].y = _3d_points[i][1] / 1000.0;
        point_clouds.points[i].z = _3d_points[i][2] / 1000.0;
        point_clouds.channels[0].values[i] = 255;
    }

    clouds_pub_.publish(point_clouds);
}
