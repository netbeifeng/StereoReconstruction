#ifndef _FEATUREDETECTOR_H
#define _FEATUREDETECTOR_H
#include <chrono>
#include "Utils.h"

enum FeatureDectectorMethod {
    USE_SIFT,
    USE_ORB,
    USE_SURF,
    USE_BRISK,
    USE_AKAZE,
    USE_KAZE
};

class FeatureDectector {
public:
    FeatureDectector(int numPoint);
    std::pair<KeyPoints,KeyPoints> findCorrespondences(ImagePair imgPairm, FeatureDectectorMethod fm);

private:
    int m_numPoint;
    cv::Ptr<cv::SIFT> SIFTDetector;
    cv::Ptr<cv::ORB> ORBDetector;
    cv::Ptr<cv::xfeatures2d::SURF> SURFDetector;
    cv::Ptr<cv::AKAZE> AKAZEDetector;
    cv::Ptr<cv::KAZE> KAZEDetector;
    cv::Ptr<cv::BRISK> BRISKDetector;

    std::string getCurrentDetectorName(FeatureDectectorMethod method) {
        switch (method) {
            case USE_SIFT: return "SIFT";
            case USE_SURF: return "SURF";
            case USE_ORB: return "ORB";
            case USE_AKAZE: return "AKAZE";
            case USE_KAZE: return "KAZE";
            case USE_BRISK: return "BRISK";
        }
        return "ERROR";
    }
};

#endif