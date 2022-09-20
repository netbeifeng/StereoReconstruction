#ifndef _EIGHTPOINT_H
#define _EIGHTPOINT_H

#include "Utils.h"

/**
 * EightPointExecuter
 * @brief Eight Point Algorithm for calculating and validating R and T
 * also includes OpenCV built-in Five Point Algorithm 
 */
class EightPointExecuter {

public:
	EightPointExecuter(std::pair<KeyPoints, KeyPoints> setPair, ImagePair sample);

	std::pair<Rotate,Translate> getValidRT();
	std::vector<std::pair<Rotate, Translate>> getAllPossibleRT();
	std::pair<bool, int> isValidRT(std::pair<Rotate, Translate> RTPair, int runnerCounter);

	cv::Mat getEssentialMatrix();
	cv::Mat getFundamentalMatrix();
	std::pair<cv::Mat, double>getLambdaGamma();
	std::pair<Rotate, Translate> executeFivePointAlgorithm();

private:
	KeyPoints m_pointSet1;
	KeyPoints m_pointSet2;

	ImagePair m_sample;
	int m_numPoint;
	double m_gamma;
	cv::Mat m_lambda;

	cv::Mat m_fundamentalMatrix;
	cv::Mat m_essentialMatrix;

	std::vector<std::pair<Rotate, Translate>> m_transformations;

	void initPossibleRT();
};
#endif 
