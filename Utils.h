#pragma once
#ifndef _UTILS_H
#define _UTILS_H

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "Eigen.h"


#define SPARSE_MATCHING 1 // Include SPARSE MATCHING in Pipeline
#define DENSE_MATCHING  1 // Include DENSE MATCHING in Pipeline

// Change here to switch off different work mode
//#define ALL_SAMPLE
#define SPEC_SAMPLE   
//#define RANDOM_SAMPLE

#define USE_MIDDLEBURY_2014

// Toggle to open optimization report
#define OPTIMIZATION_LOG_VERBOSE 0

typedef std::vector<cv::Point2f> KeyPoints;
typedef cv::Mat Rotate;
typedef cv::Mat Translate;

// Structure for representationg of an image pair
struct ImagePair {
    std::string path;

    cv::Mat img1;
    cv::Mat img2;

    cv::Mat K_img1;
    cv::Mat K_img2;

    float f1;
    float f2;

    float baseline;
    float doffs;

    int width;
    int height;

    int vmin;
    int vmax;

    std::string name;
    /*
        Ground truth disperity of left image.
    */
    cv::Mat disp0;

    /*
        Ground truth disperity of right image.
    */
    cv::Mat disp1;

    // Returns a new ImagePair with the same information/matrices as the
    // original one with the difference that it is downsampled by `factor`
    ImagePair sampleDown(float factor) {
        cv::Mat img1_d;
        cv::Mat img2_d;
        cv::Mat disp0_d;
        cv::Mat disp1_d;

        cv::resize(img1, img1_d, cv::Size(factor * img1.cols, factor * img1.rows),
            0, 0, cv::INTER_LINEAR);
        cv::resize(img2, img2_d, cv::Size(factor * img2.cols, factor * img2.rows),
            0, 0, cv::INTER_LINEAR);
        cv::resize(disp0, disp0_d, cv::Size(factor * disp0.cols, factor * disp0.rows),
            0, 0, cv::INTER_LINEAR);
        cv::resize(disp1, disp1_d, cv::Size(factor * disp1.cols, factor * disp1.rows),
            0, 0, cv::INTER_LINEAR);

        ImagePair new_pair{};

        new_pair.img1 = img1_d;
        new_pair.img2 = img2_d;
        new_pair.K_img1 = K_img1;
        new_pair.K_img2 = K_img2;
        new_pair.disp0 = disp0_d;
        new_pair.disp1 = disp1_d;
        new_pair.f1 = f1;
        new_pair.f2 = f2;
        new_pair.baseline = baseline;
        new_pair.doffs = doffs;
        new_pair.width = width;
        new_pair.height = height;
        new_pair.vmin = vmin;
        new_pair.vmax = vmax;

        return new_pair;
    }
};

// sorting DMatch by its distance
inline bool distanceSorting(cv::DMatch a, cv::DMatch b) {
    return a.distance < b.distance;
}

// \hat{w}
inline cv::Mat makeSkewMatrixFromPoint(cv::Point3f p) {
    cv::Mat skewMatrix = (cv::Mat_<double>(3, 3) <<    0,  -1,  p.y, 
                                                       1,   0, -p.x,
                                                    -p.y, p.x,   0);

    return skewMatrix;
}

inline cv::Mat getEulerAngleByRotationMatrix(cv::Mat Rotate) {
    double R32 = Rotate.at<double>(2, 1);
    double R33 = Rotate.at<double>(2, 2);
    double thetaX = std::atan2(R32, R33);
    double thetaY = std::atan2(-Rotate.at<double>(2, 0), std::sqrt(R32 * R32 + R33 * R33));
    double thetaZ = std::atan2(Rotate.at<double>(1, 0), Rotate.at<double>(0, 0));

    cv::Mat retM = (cv::Mat_<double>(3, 1) << thetaX, thetaY, thetaZ);

    return retM;
}

inline cv::Mat getRoationMatrixByEulerAngle(cv::Mat angle) {
    double thetaX = angle.at<double>(0, 0);
    double thetaY = angle.at<double>(1, 0);
    double thetaZ = angle.at<double>(2, 0);

    cv::Mat X = (cv::Mat_<double>(3, 3) << 1, 0, 0,
        0, std::cos(thetaX), -std::sin(thetaX),
        0, std::sin(thetaX), std::cos(thetaX));

    cv::Mat Y = (cv::Mat_<double>(3, 3) << std::cos(thetaY), 0, std::sin(thetaY),
        0, 1, 0,
        -std::sin(thetaY), 0, std::cos(thetaY));

    cv::Mat Z = (cv::Mat_<double>(3, 3) << std::cos(thetaZ), -std::sin(thetaZ), 0,
        std::sin(thetaZ), std::cos(thetaZ), 0,
        0, 0, 1);

    return Z * Y * X;
}

// Checks if a matrix is a valid rotation matrix.
inline bool isRotationMatrix(cv::Mat& R)
{
	cv::Mat Rt;
	cv::transpose(R, Rt);
	cv::Mat shouldBeIdentity = Rt * R;
	cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());

	return  cv::norm(I, shouldBeIdentity) < 1e-6;

}

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
inline cv::Vec3f rotationMatrixToEulerAngles(cv::Mat& R)
{

	assert(isRotationMatrix(R));

	float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

	bool singular = sy < 1e-6; // If

	float x, y, z;
	if (!singular)
	{
		x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
		y = atan2(-R.at<double>(2, 0), sy);
		z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
	}
	else
	{
		x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
		y = atan2(-R.at<double>(2, 0), sy);
		z = 0;
	}
	return cv::Vec3f(x, y, z);

}

// Evaluation function for translation 
inline float evaluateTranslation(float baseline, Translate t) {
    double baselineScaled = baseline * 0.001; // 0.001 here for rescaling in meters
    Translate scaledT = t * baselineScaled;
    cv::Mat groundTruth = (cv::Mat_<double>(3, 1) << baselineScaled, 0.0, 0);
    return cv::norm(-scaledT - groundTruth);
}

// Evaluation function for rotation 
inline float evaluateRotation(Rotate r) {
    return cv::norm(rotationMatrixToEulerAngles(r)); // GT should be 0
}

#endif