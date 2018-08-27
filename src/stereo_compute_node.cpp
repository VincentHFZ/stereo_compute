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

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "stereo_compute");
    ros::NodeHandle nh;

    Mat srcImage_1 = imread("/home/eventc/Desktop/DeepCascadePedestrainDetection/data/sample_test_images/imgL_00.png");
    Mat srcImage_2 = imread("/home/eventc/Desktop/DeepCascadePedestrainDetection/data/sample_test_images/imgR_00.png");

    vector<KeyPoint> key_points_1, key_points_2;
    Mat descriptors_1, descriptors_2;
    vector<DMatch> matches;
    // vector<DMatch> dst_l, dst_r;
    vector<Point2f> point2f_l, point2f_r;
    vector<Point2f> dst_l, dst_r;

    cv::Mat_<double> camParamL, camParamR;
    cv::Mat_<double> distParamL, distParamR;
    cv::Mat_<double> matrixR, matrixT;

    if(srcImage_1.empty() || srcImage_2.empty())
    {
        if(srcImage_1.empty())
            cout << "Load image 1 failed!" << endl;
        else if(srcImage_2.empty())
            cout << "Load image 2 failed!" << endl;

        return -1;
    }

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

    cv::Mat R1, R2, P1, P2, Q;
    cv::Size imgSize(640, 480);

    cv::Mat mapLx, mapLy, mapRx, mapRy;
    cv::Rect validROIL, validROIR;
    stereoRectify(camParamL, distParamL,
                  camParamR, distParamR,
                  imgSize, matrixR, matrixT, R1, R2, P1, P2, Q,
                  cv::CALIB_ZERO_DISPARITY, 0, imgSize, &validROIL, &validROIR);
    initUndistortRectifyMap(camParamL, distParamL, R1, P1, imgSize, CV_32FC1, mapLx, mapLy);
    initUndistortRectifyMap(camParamR, distParamR, R2, P2, imgSize, CV_32FC1, mapRx, mapRy);

    remap(srcImage_1.clone(), srcImage_1, mapLx, mapLy, cv::INTER_LINEAR);
    remap(srcImage_2.clone(), srcImage_2, mapRx, mapRy, cv::INTER_LINEAR);

    cv::SiftFeatureDetector detector(800);        //定义特点点检测器
    detector.detect(srcImage_1, key_points_1);
    detector.detect(srcImage_2, key_points_2);

    Mat outImage_1, outImage_2;
    drawKeypoints(srcImage_1, key_points_1, descriptors_1);
    drawKeypoints(srcImage_2, key_points_2, descriptors_2);

    // 提取特征向量
    cv::SiftDescriptorExtractor extractor;
    extractor.compute(srcImage_1, key_points_1, descriptors_1);
    extractor.compute(srcImage_2, key_points_2, descriptors_2);

//    cv::SurfFeatureDetector detector(800);        //定义特点点检测器
//    detector.detect(srcImage_1, key_points_1);
//    detector.detect(srcImage_2, key_points_2);

//    Mat outImage_1, outImage_2;
//    drawKeypoints(srcImage_1, key_points_1, descriptors_1);
//    drawKeypoints(srcImage_2, key_points_2, descriptors_2);

//    // 提取特征向量
//    cv::SurfDescriptorExtractor extractor;
//    extractor.compute(srcImage_1, key_points_1, descriptors_1);
//    extractor.compute(srcImage_2, key_points_2, descriptors_2);

    // use ORB
//    ORB orb;
//    orb.detect(srcImage_1, key_points_1);
//    orb.detect(srcImage_2, key_points_2);
//    orb.compute(srcImage_1, key_points_1, descriptors_1);
//    orb.compute(srcImage_2, key_points_2, descriptors_2);


    // 匹配特征点，主要计算两个特征点特征向量的欧式距离，距离小于某个阈值则认为匹配
    BruteForceMatcher<L2<float> > matcher;
    matcher.match(descriptors_1, descriptors_2, matches);

    Mat img_matches;
    drawMatches(srcImage_1, key_points_1,
                srcImage_2, key_points_2,
                matches,
                img_matches);

    // RANSAC
    // Step1 根据 matches 将特征点对齐，将坐标转为为 float 类型
    vector<KeyPoint> R_keypoints_1;
    vector<KeyPoint> R_keypoints_2;
    for(size_t i = 0; i < matches.size(); i++)
    {
        R_keypoints_1.push_back(key_points_1[matches[i].queryIdx]);
        R_keypoints_2.push_back(key_points_2[matches[i].trainIdx]);
    }
    // 坐标转换
    vector<Point2f> p01, p02;
    for(size_t i = 0; i < matches.size(); i++)
    {
        p01.push_back(R_keypoints_1[i].pt);
        p02.push_back(R_keypoints_2[i].pt);
    }
    // 利用基础矩阵剔除误匹配点
    vector<uchar> RansacStatus;
    // Mat Fundamental= findFundamentalMat(p01, p02, RansacStatus, FM_RANSAC); // 基础
    Mat Homography = findHomography(p01, p02, RansacStatus, FM_RANSAC);

    // 重新定义 RR_keypoint 和 RR_matches 来存储新的关键点和匹配矩阵
    vector<KeyPoint> RR_keypoints_1, RR_keypoints_2;
    vector<DMatch> RR_matches;
    int iidex = 0;
    for(size_t i = 0; i < matches.size(); i++)
    {
        if(RansacStatus[i] != 0)
        {
            RR_keypoints_1.push_back(R_keypoints_1[i]);
            RR_keypoints_2.push_back(R_keypoints_2[i]);
            matches[i].queryIdx = iidex;
            matches[i].trainIdx = iidex;
            RR_matches.push_back(matches[i]);
            iidex++;
        }
    }
    Mat img_RR_matches;
    drawMatches(srcImage_1, RR_keypoints_1,
                srcImage_2, RR_keypoints_2,
                RR_matches, img_RR_matches);
    imshow("result", img_RR_matches);

    for(size_t i = 0; i < RR_keypoints_1.size(); i++)
    {
        point2f_l.push_back(RR_keypoints_1[i].pt);
        point2f_r.push_back(RR_keypoints_2[i].pt);
    }

    std::cout << "!" << std::endl;
    // 立体解算


//    stereoRectify(camParamL, distParamL,
//                  camParamR, distParamR,
//                  imgSize,
//                  matrixR, matrixT,
//                  R1, R2,
//                  P1, P2, Q);
//    undistortPoints(point2f_l, dst_l, camParamL, distParamL, R1, P1);
//    undistortPoints(point2f_r, dst_r, camParamR, distParamR, R2, P2);

    Matx44f _Q;
    Q.convertTo(_Q, CV_64F);
    cv::Vec3f _3d_point;
    vector<Vec3f> _3d_points;

    std::cout << "@" << std::endl;

//    for(size_t i = 0; i < dst_l.size(); i++)
//    {
//        double d = dst_l[i].x - dst_r[i].x;
//        Vec4f homg_pt = _Q * Vec4f(dst_l[i].x, dst_l[i].y, d, 1);

//        _3d_point = Vec3f(homg_pt.val);
//        _3d_point /= homg_pt[3];
//        //_3d_point /= 1000.0;

//        std::cout << "_3dPoints: \t"
//                  << "x: " << _3d_point[0] << "\t"
//                  << "y: " << _3d_point[1] << "\t"
//                  << "z: " << _3d_point[2] << "\t"
//                  << std::endl;
//        _3d_points.push_back(_3d_point);
//    }

    for(size_t i = 0; i < point2f_l.size(); i++)
    {
        double d = point2f_l[i].x - point2f_r[i].x;
        Vec4f homg_pt = _Q * Vec4f(point2f_l[i].x, point2f_l[i].y, d, 1);

        _3d_point = Vec3f(homg_pt.val);
        _3d_point /= homg_pt[3];
        //_3d_point /= 1000.0;

        std::cout << "_3dPoints: \t"
                  << "x: " << _3d_point[0] << "\t"
                  << "y: " << _3d_point[1] << "\t"
                  << "z: " << _3d_point[2] << "\t"
                  << std::endl;
        _3d_points.push_back(_3d_point);
    }

    sensor_msgs::PointCloud point_clouds;
    point_clouds.header.frame_id = "camera_link";
    point_clouds.header.stamp = ros::Time::now();
    point_clouds.points.resize(_3d_points.size());

    point_clouds.channels.resize(1);
    point_clouds.channels[0].name = "rgb";
    point_clouds.channels[0].values.resize(_3d_points.size());

    for(size_t i = 0; i < _3d_points.size(); i++)
    {
        point_clouds.points[i].x = _3d_points[i][0] / 1000;
        point_clouds.points[i].y = _3d_points[i][1] / 1000;
        point_clouds.points[i].z = _3d_points[i][2] / 1000;
        point_clouds.channels[0].values[i] = 255;
    }


    ros::Rate rate(20);

    ros::Publisher cloud_pub = nh.advertise<sensor_msgs::PointCloud>("cloud", 50);

    while(ros::ok())
    {
        cloud_pub.publish(point_clouds);
        waitKey(1);
        rate.sleep();
    }

    return 0;
}
