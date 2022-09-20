#include "FeatureDectector.h"

FeatureDectector::FeatureDectector(int numPoint) {
    m_numPoint = numPoint;
}

std::pair<KeyPoints,KeyPoints> FeatureDectector::findCorrespondences(ImagePair imgPair, FeatureDectectorMethod fm) {
    cv::Mat img1;
    cv::Mat img2;
  
    cv::cvtColor(imgPair.img1, img1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgPair.img2, img2, cv::COLOR_BGR2GRAY);


    if (img1.empty() || img2.empty()) {
        throw std::length_error("FeatureDectector >> Fail to Load the Image.");
    }
    else {
        std::cerr << "FeatureDectector >> Images successfully loaded." << std::endl;
    }

    if (fm == USE_SIFT) {
        // https://docs.opencv.org/3.4/d7/d60/classcv_1_1SIFT.html
        SIFTDetector = cv::SIFT::create(0, 4, 0.04, 10, 1.6); 
        std::cout << "FeatureDectector >> SIFTDectector loaded." << std::endl;
    } else if (fm == USE_ORB) {
        // https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html
        ORBDetector = cv::ORB::create();
        std::cout << "FeatureDectector >> ORBDectector loaded." << std::endl;
    } else if (fm == USE_SURF) {
        // https://docs.opencv.org/3.4/d5/df7/classcv_1_1xfeatures2d_1_1SURF.html
        SURFDetector = cv::xfeatures2d::SURF::create(100, 4, 4, false, false);
        std::cout << "FeatureDectector >> SURFDectector loaded." << std::endl;
    } else if (fm == USE_BRISK) {
        // https://docs.opencv.org/3.4/de/dbf/classcv_1_1BRISK.html
        BRISKDetector = cv::BRISK::create(30, 4 , 1.0f);
        std::cout << "FeatureDectector >> BRISKDectector loaded." << std::endl;
    } else if (fm == USE_AKAZE) {
        // https://docs.opencv.org/3.4/d8/d30/classcv_1_1AKAZE.html
        AKAZEDetector = cv::AKAZE::create();
        std::cout << "FeatureDectector >> AKAZEDectector loaded." << std::endl;
    } else if (fm == USE_KAZE) {
        // https://docs.opencv.org/4.x/d3/d61/classcv_1_1KAZE.html
        KAZEDetector = cv::KAZE::create();
        std::cout << "FeatureDectector >> KAZEDectector loaded." << std::endl;
    }

    std::vector<cv::KeyPoint> k_points1, k_points2;
    cv::Mat descr1, descr2;

    auto start = std::chrono::steady_clock::now();

    if (fm == USE_SIFT) {
        SIFTDetector.get()->detectAndCompute(img1, cv::noArray(), k_points1, descr1);
        SIFTDetector.get()->detectAndCompute(img2, cv::noArray(), k_points2, descr2);
    } else if (fm == USE_ORB) {
        ORBDetector.get()->detectAndCompute(img1, cv::noArray(), k_points1, descr1);
        ORBDetector.get()->detectAndCompute(img2, cv::noArray(), k_points2, descr2);
    } else if (fm == USE_SURF) {
        SURFDetector.get()->detectAndCompute(img1, cv::noArray(), k_points1, descr1);
        SURFDetector.get()->detectAndCompute(img2, cv::noArray(), k_points2, descr2);
    } else if (fm == USE_BRISK) {
        BRISKDetector.get()->detectAndCompute(img1, cv::noArray(), k_points1, descr1);
        BRISKDetector.get()->detectAndCompute(img2, cv::noArray(), k_points2, descr2);
    } else if (fm == USE_AKAZE) {
        AKAZEDetector.get()->detectAndCompute(img1, cv::noArray(), k_points1, descr1);
        AKAZEDetector.get()->detectAndCompute(img2, cv::noArray(), k_points2, descr2);
    } else if (fm == USE_KAZE) {
        KAZEDetector.get()->detectAndCompute(img1, cv::noArray(), k_points1, descr1);
        KAZEDetector.get()->detectAndCompute(img2, cv::noArray(), k_points2, descr2);
    }

    // Necessary, convert to 32F required by FlannBasedMatcher
    descr1.convertTo(descr1, CV_32F);
    descr2.convertTo(descr2, CV_32F);

    std::vector<cv::DMatch> matches;
    std::vector<cv::DMatch> good_matches;
    
    cv::FlannBasedMatcher matcher;
    matcher.match(descr1, descr2, matches);

    int foundMatches = matches.size();

    auto end = std::chrono::steady_clock::now();

    // Sorting the distance to get best N
    std::sort(matches.begin(), matches.end(), distanceSorting);

    KeyPoints image1Points;
    KeyPoints image2Points;

    if (matches.size() < m_numPoint) {
        std::cerr << "FeatureDetector >> Error didn't get enough feature points." << std::endl;
        throw std::bad_exception();
    } else {
        for (int i = 0; i < m_numPoint; i++) {
            cv::KeyPoint keyPoint1 = k_points1.at(matches[i].queryIdx);
            cv::Point2f point1 = cv::Point2f(keyPoint1.pt.x, keyPoint1.pt.y);

            cv::KeyPoint keyPoint2 = k_points2.at(matches[i].trainIdx);
            cv::Point2f point2 = cv::Point2f(keyPoint2.pt.x, keyPoint2.pt.y);
            // Filling the result vector
            image1Points.push_back(point1);
            image2Points.push_back(point2);
            good_matches.push_back(matches.at(i));
        }
    }

    cv::Mat m_img;
    cv::drawMatches(imgPair.img1, k_points1, imgPair.img2,
                    k_points2, good_matches, m_img,
                    cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    // Write the matching result to dir
    cv::imwrite(imgPair.name + "_" + getCurrentDetectorName(fm) + ".png", m_img);

    float sumDis = 0;
    for (int i = 0; i < m_numPoint; i++)
    {
        std::printf("FeatureDectector >> Good Match [%d] Keypoint 1: %d = = = Keypoint 2: %d , DIS = %f.\n", i, matches[i].queryIdx, matches[i].trainIdx, matches[i].distance);
        sumDis += good_matches[i].distance;
    }
    sumDis /= m_numPoint;
    std::cout << "FeatureDectector >> Dis: " << sumDis << std::endl;
    std::cout << "FeatureDectector >> Found matches: " << foundMatches << std::endl;
    std::cout << "FeatureDectector >> Elapsed time in seconds: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
        << " ms" << std::endl;

    return std::make_pair(image1Points, image2Points);
}
