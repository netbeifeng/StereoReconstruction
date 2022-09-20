#ifndef _BLOCKMATCHER_H
#define _BLOCKMATCHER_H

#include "Utils.h"

enum BlockMatcherMethod {
    USE_BM,
    USE_SGBM
};

enum MatchingMethod {
    OPENCV_BM,
    OPENCV_SGBM,
    SAD
};

enum EvaluationMetric {
    RMS,
    BAD_1,
    BAD_2,
    BAD_5
};

class BlockMatcher {
public:
    BlockMatcher(ImagePair imgPair);
    cv::Mat execute(MatchingMethod method, int block_size, int num_disp);
    float evaluate(cv::Mat ground_truth, cv::Mat computed, EvaluationMetric metric);
    void evaluateSGBM(int block_size, int num_disp);
    void evaluateBM(int block_size, int num_disp);

private:
    ImagePair m_imgPair;

    cv::Mat sad(cv::Mat left, cv::Mat right, int window_size) {
        int width = left.size().width;
        int height = right.size().height;
        int max_offset = 79;

        cv::Mat depth(height, width, 0);
        std::vector<std::vector<int> > min_ssd; // store min SSD values

        for (int i = 0; i < height; ++i) {
            std::vector<int> tmp(width, std::numeric_limits<int>::max());
            min_ssd.push_back(tmp);
        }

        for (int offset = 0; offset <= max_offset; offset++) {
            cv::Mat tmp(height, width, 0);
            // shift image depend on type to save calculation time

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < offset; x++) {
                    tmp.at<uchar>(y, x) = right.at<uchar>(y, x);
                }

                for (int x = offset; x < width; x++) {
                    tmp.at<uchar>(y, x) = right.at<uchar>(y, x - offset);
                }
            }

            // calculate each pixel's SSD value
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int start_x = std::max(0, x - window_size);
                    int start_y = std::max(0, y - window_size);
                    int end_x = std::min(width - 1, x + window_size);
                    int end_y = std::min(height - 1, y + window_size);
                    int sum_sd = 0;

                    for (int i = start_y; i <= end_y; i++) {
                        for (int j = start_x; j <= end_x; j++) {
                            int delta = abs(left.at<uchar>(i, j) - tmp.at<uchar>(i, j));
                            sum_sd += delta * delta;
                        }
                    }

                    // smaller SSD value found
                    if (sum_sd < min_ssd[y][x]) {
                        min_ssd[y][x] = sum_sd;
                        // for better visualization
                        depth.at<uchar>(y, x) = (uchar)(offset * 3);
                    }
                }
            }
        }

        return depth;
    }

};
#endif // !_BLOCKMATCHER_H
