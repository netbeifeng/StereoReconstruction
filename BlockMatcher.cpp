#include "BlockMatcher.h"
#include "Reconstructor.hpp"

BlockMatcher::BlockMatcher(ImagePair imgPair) {
    m_imgPair = imgPair;
}

// Evaluation function for BM/SGBM
float BlockMatcher::evaluate(cv::Mat ground_truth, cv::Mat computed, EvaluationMetric metric) {
    if (ground_truth.size() != computed.size()) {
        std::cerr << "BlockMatcher >> Mismatch in disperity map size\n";
        return -1.0f;
    }

    int dist = 1;

    switch (metric) {
    case EvaluationMetric::BAD_1: dist = 1; break;
    case EvaluationMetric::BAD_2: dist = 2; break;
    case EvaluationMetric::BAD_5: dist = 5; break;
    }

    switch (metric) {
    case RMS: {
        float abs_error = 0;
        for (int y = 0; y < ground_truth.rows; y++) {
            for (int x = 0; x < ground_truth.cols; x++) {
                float diff = computed.at<unsigned char>(y, x) - ground_truth.at<unsigned char>(y, x);
                abs_error += diff * diff;
            }
        }

        return std::sqrt(abs_error / (float)(ground_truth.cols * ground_truth.rows));
    }
    case BAD_1:
    case BAD_2:
    case BAD_5: {
        int width = ground_truth.cols;
        int height = ground_truth.rows;
        int bad_pixels = 0;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int delta = std::abs(computed.at<unsigned char>(y, x) - ground_truth.at<unsigned char>(y, x));
                if (delta > dist && computed.at<unsigned char>(y, x) != 0) {
                    bad_pixels++;
                }
            }
        }

        float rate_left = float(bad_pixels) / (float)(height * width);
        return rate_left * 100;
    }
    default:
        return 0.0;
    }
}

// Execute BM or SGBM
cv::Mat BlockMatcher::execute(MatchingMethod method, int block_size, int num_disp) {
    cv::Mat img1 = m_imgPair.img1;
    cv::Mat img2 = m_imgPair.img2;
    // Convert input images to grayscale
    // This makes matching better and doesn't change the matched windows
    cv::Mat img1_gray{};
    cv::Mat img2_gray{};

    cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);

    switch (method) {
    case MatchingMethod::OPENCV_BM: {
        cv::Mat disparity;
        cv::Ptr<cv::StereoMatcher> matcher = cv::StereoBM::create(num_disp, block_size);
        matcher->compute(img1_gray, img2_gray, disparity);

        // Divide by 16 because OpenCV disperity map is calculated with 4 bit of fractional precision
        disparity = disparity / 16;
        disparity.convertTo(disparity, CV_8U);
        return disparity;
    }

    case MatchingMethod::OPENCV_SGBM: {
        cv::Mat disparity;
        cv::Ptr<cv::StereoSGBM> matcher = cv::StereoSGBM::create(0, num_disp, block_size);
        matcher->setP1(24 * block_size * block_size);
        matcher->setP2(96 * block_size * block_size);
        matcher->compute(img1_gray, img2_gray, disparity);

        // Divide by 16 because OpenCV disperity map is calculated with 4 bit of fractional precision
        disparity = disparity / 16;
        disparity.convertTo(disparity, CV_8U);
        return disparity;
    }
    case MatchingMethod::SAD: {
        return sad(img1_gray, img2_gray, block_size);
    }
    default:
        throw std::invalid_argument("BlockMatcher >> Unsupported matching algorithm");
    }

    cv::Mat dummy{};
    return dummy;
}

void BlockMatcher::evaluateSGBM(int block_size, int num_disp) {
    Reconstructor pc{};
    float max_depth = 0.0;

    // Compute disparity map using SGBM
    cv::Mat disp_sgbm = execute(MatchingMethod::OPENCV_SGBM, block_size, num_disp);

    float bad1_sgbm = evaluate(m_imgPair.disp0, disp_sgbm, EvaluationMetric::BAD_1);
    float bad2_sgbm = evaluate(m_imgPair.disp0, disp_sgbm, EvaluationMetric::BAD_2);
    float bad5_sgbm = evaluate(m_imgPair.disp0, disp_sgbm, EvaluationMetric::BAD_5);
    float rms_sgbm = evaluate(m_imgPair.disp0, disp_sgbm, EvaluationMetric::RMS);
    std::cout << "BlockMatcher >> Metrics - SGBM\n";
    std::cout << "BlockMatcher >> BAD-1 " << 100.0 - bad1_sgbm << "\n";
    std::cout << "BlockMatcher >> BAD-2 " << 100.0 - bad2_sgbm << "\n";
    std::cout << "BlockMatcher >> BAD-5 " << 100.0 - bad5_sgbm << "\n";
    std::cout << "BlockMatcher >> RMS " << rms_sgbm << "\n";

    cv::Mat depth_sgbm = pc.depthMapFromDisperityMap(disp_sgbm, m_imgPair.baseline, m_imgPair.doffs, m_imgPair.f1, &max_depth, true);

    std::stringstream ss;
    ss << m_imgPair.name + "_sgbm.off";
    Vertex* vertices = pc.generatePointCloud(depth_sgbm, m_imgPair.img1, m_imgPair.K_img1, max_depth);

    if (!Mesh::writeMesh(vertices, depth_sgbm.cols, depth_sgbm.rows, ss.str())) {
        std::cerr << "BlockMatcher >> Could not write mesh " << ss.str() << "\n";
    }

    delete[] vertices;

    //cv::resize(disp_sgbm, disp_sgbm, cv::Size(0.4 * disp_sgbm.cols, 0.4 * disp_sgbm.rows), 0, 0, cv::INTER_LINEAR);
    //cv::resize(depth_sgbm, depth_sgbm, cv::Size(0.4 * disp_sgbm.cols, 0.4 * disp_sgbm.rows), 0, 0, cv::INTER_LINEAR);

    depth_sgbm = 255 * depth_sgbm;
    depth_sgbm.convertTo(depth_sgbm, CV_8UC1);

    cv::imwrite(m_imgPair.name + "_disp_normal_sgbm.png", disp_sgbm);

    cv::Mat diff = m_imgPair.disp0 - disp_sgbm;
    cv::Mat error;
    cv::applyColorMap(diff, error, cv::COLORMAP_JET);
    cv::imwrite(m_imgPair.name + "_disp_normal_sgbm_error.png", error);
    std::cout << "BlockMatcher >> SGBM Error: " << cv::mean(diff)[0] << std::endl;

    cv::applyColorMap(disp_sgbm, disp_sgbm, cv::COLORMAP_JET);
    cv::imwrite(m_imgPair.name + "_disp_jet_sgbm.png", disp_sgbm);

    cv::imwrite(m_imgPair.name + "_depth_sgbm.png", depth_sgbm);
}

void BlockMatcher::evaluateBM(int block_size, int num_disp) {
    Reconstructor pc{};
    float max_depth = 0.0;

    // Compute disparity map using SGBM
    cv::Mat disp_bm = execute(MatchingMethod::OPENCV_BM, block_size, num_disp);

    float bad1_bm = evaluate(m_imgPair.disp0, disp_bm, EvaluationMetric::BAD_1);
    float bad2_bm = evaluate(m_imgPair.disp0, disp_bm, EvaluationMetric::BAD_2);
    float bad5_bm = evaluate(m_imgPair.disp0, disp_bm, EvaluationMetric::BAD_5);
    float rms_bm = evaluate(m_imgPair.disp0, disp_bm, EvaluationMetric::RMS);
    std::cout << "BlockMatcher >> Metrics - BM\n";
    std::cout << "BlockMatcher >> BAD-1 " << 100.0 - bad1_bm << "\n";
    std::cout << "BlockMatcher >> BAD-2 " << 100.0 - bad2_bm << "\n";
    std::cout << "BlockMatcher >> BAD-5 " << 100.0 - bad5_bm << "\n";
    std::cout << "BlockMatcher >> RMS " << rms_bm << "\n";

    cv::Mat depth_bm = pc.depthMapFromDisperityMap(disp_bm, m_imgPair.baseline, m_imgPair.doffs, m_imgPair.f1, &max_depth, true);

    std::stringstream ss;
    ss << m_imgPair.name + "_bm.off";

    Vertex* vertices = pc.generatePointCloud(depth_bm, m_imgPair.img1, m_imgPair.K_img1, max_depth);

    if (!Mesh::writeMesh(vertices, depth_bm.cols, depth_bm.rows, ss.str())) {
        std::cerr << "BlockMatcher >> Could not write mesh " << ss.str() << "\n";
    }

    delete[] vertices;

    //cv::resize(disp_bm, disp_bm, cv::Size(0.4 * disp_bm.cols, 0.4 * disp_bm.rows), 0, 0, cv::INTER_LINEAR);
    //cv::resize(depth_bm, depth_bm, cv::Size(0.4 * depth_bm.cols, 0.4 * depth_bm.rows), 0, 0, cv::INTER_LINEAR);

    depth_bm = 255 * depth_bm;
    depth_bm.convertTo(depth_bm, CV_8UC1);

    cv::imwrite(m_imgPair.name + "_disp_normal_bm.png", disp_bm);

    cv::Mat diff = m_imgPair.disp0 - disp_bm;
    cv::Mat error;
    cv::applyColorMap(diff, error, cv::COLORMAP_JET);
    cv::imwrite(m_imgPair.name + "_disp_normal_bm_error.png", error);
    std::cout <<  "BlockMatcher >> BM Error: " << cv::mean(diff)[0] << std::endl;

    cv::applyColorMap(disp_bm, disp_bm, cv::COLORMAP_JET);
    cv::imwrite(m_imgPair.name + "_disp_jet_bm.png", disp_bm);

    cv::imwrite(m_imgPair.name + "_depth_bm.png", depth_bm);
}
