#include "FeatureDectector.h"
#include "Dataloader.hpp"
#include "Reconstructor.hpp"
#include "EightPoint.h"
#include "BAOptimizer.h"
#include "BlockMatcher.h"
#include "Utils.h"

// Include the Pose recovery as well as the Bundle Adjustment Optimization
void calculateAndOptimizePose(ImagePair pair, std::pair<KeyPoints, KeyPoints> points){
	EightPointExecuter eightPointExecuter = EightPointExecuter(points, pair);
	std::pair<Rotate, Translate> validRT = eightPointExecuter.getValidRT();
	validRT = eightPointExecuter.executeFivePointAlgorithm();

	Rotate R = validRT.first;
	Translate T = validRT.second * pair.baseline * 0.001;
	validRT = std::make_pair(R,T);

	std::cout << "MAIN >> -------- BEFORE --------" << std::endl;
	std::cout << "MAIN >> Error in translation :" << evaluateTranslation(pair.baseline, T) << std::endl;
	std::cout << "MAIN >> Error in rotation: " << evaluateRotation(R) << std::endl;

	BAOptimizer baOptimizer = BAOptimizer(pair, points);
	std::cout << "MAIN >> -------- AFTER --------" << std::endl;

	// Optimization, 50 means iteration number 
	std::pair<Rotate, Translate> optimizedRT = baOptimizer.optimize(validRT, 50);
	R = optimizedRT.first;
	T = optimizedRT.second;
	
	std::cout << "MAIN >> Error in translation :" << evaluateTranslation(pair.baseline, T) << std::endl;
	std::cout << "MAIN >> Error in rotation: " << evaluateRotation(R) << std::endl;
}

void sparseMatchingForImagePair(ImagePair specSample, FeatureDectector detector) {
		std::cout << "DataLoader >> " << specSample.path << " is selected." << std::endl;
		std::pair<KeyPoints, KeyPoints> corrPointsORB = detector.findCorrespondences(specSample, USE_ORB);
		std::pair<KeyPoints, KeyPoints> corrPointsSIFT = detector.findCorrespondences(specSample, USE_SIFT);
		std::pair<KeyPoints, KeyPoints> corrPointsSURF = detector.findCorrespondences(specSample, USE_SURF);
		std::pair<KeyPoints, KeyPoints> corrPointsBRISK = detector.findCorrespondences(specSample, USE_BRISK);
		std::pair<KeyPoints, KeyPoints> corrPointsAKAZE = detector.findCorrespondences(specSample, USE_AKAZE);
		std::pair<KeyPoints, KeyPoints> corrPointsKAZE = detector.findCorrespondences(specSample, USE_KAZE);

		std::cout << std::endl;

		std::cout << "MAIN >> --- ORB Detector Pose Start ---" << std::endl;
		calculateAndOptimizePose(specSample, corrPointsORB); // ORB
		std::cout << "MAIN >> --- ORB Detector Pose End ---" << std::endl;
		std::cout << std::endl;

		std::cout << "MAIN >> --- SIFT Detector Pose Start ---" << std::endl;
		calculateAndOptimizePose(specSample, corrPointsSIFT); // SIFT
		std::cout << "MAIN >> --- SIFT Detector Pose End ---" << std::endl;
		std::cout << std::endl;

		std::cout << "MAIN >> --- SURF Detector Pose Start ---" << std::endl;
		calculateAndOptimizePose(specSample, corrPointsSURF); // SURF
		std::cout << "MAIN >> --- SURF Detector Pose End ---" << std::endl;
		std::cout << std::endl;

		std::cout << "MAIN >> --- BRISK Detector Pose Start ---" << std::endl;
		calculateAndOptimizePose(specSample, corrPointsBRISK); // BRISK
		std::cout << "MAIN >> --- BRISK Detector Pose End ---" << std::endl;
		std::cout << std::endl;

		std::cout << "MAIN >> --- AKAZE Detector Pose Start ---" << std::endl;
		calculateAndOptimizePose(specSample, corrPointsAKAZE); // AKAZE
		std::cout << "MAIN >> --- AKAZE Detector Pose End ---" << std::endl;
		std::cout << std::endl;

		std::cout << "MAIN >> --- KAZE Detector Pose Start ---" << std::endl;
		calculateAndOptimizePose(specSample, corrPointsKAZE); // KAZE
		std::cout << "MAIN >> --- KAZE Detector Pose End ---" << std::endl;
		std::cout << std::endl;
}

int main(int argc, char** argv) {
	// make OpenCV silent too urusai
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

	// If you want to load all
	// Use  DataLoader("Middlebury_2014") as Constructor it loads all by default
	// Warning: will consume about 2GB memory

	// Load a sepcific num of imagePair => use like 12 for 12 points
	// DataLoader dataLoader = DataLoader("Middlebury_2014", 12);

	// Load a specific imagePair => use -1
	// DataLoader dataLoader = DataLoader("Middlebury_2014", -1);
	DataLoader dataLoader = DataLoader("Middlebury_2014", "Piano");

	if (SPARSE_MATCHING) {
		FeatureDectector detector = FeatureDectector(40);
		std::cout << std::endl;

#ifdef defined(ALL_SAMPLE)
		// for each img pairs
		for (ImagePair ip : dataLoader.getAllImagePairs()) {
			sparseMatchingForImagePair(ip, detector);
		}
#elif defined(RANDOM_SAMPLE)
		ImagePair randSample = dataLoader.getRandomSample();
		sparseMatchingForImagePair(randSample, detector);
#elif defined(SPEC_SAMPLE)
		ImagePair specSample = dataLoader.getSpecificSample();
		sparseMatchingForImagePair(specSample, detector);
	} else if (DENSE_MATCHING) {
		ImagePair specSample = dataLoader.getSpecificSample();
		std::cout << "DataLoader >> " << specSample.path << " is selected." << std::endl;
		BlockMatcher blockMatcher = BlockMatcher(specSample);
		blockMatcher.evaluateSGBM(7, 260);
		blockMatcher.evaluateBM(21, 256);

		Reconstructor pc{};
		float max_depth;

		cv::Mat depth = pc.depthMapFromDisperityMap(specSample.disp0, specSample.baseline, specSample.doffs, specSample.f1, &max_depth, true);

		cv::resize(specSample.img1, specSample.img1, cv::Size(0.4 * specSample.img1.cols, 0.4 * specSample.img1.rows), 0, 0, cv::INTER_LINEAR);
		cv::resize(specSample.disp0, specSample.disp0, cv::Size(0.4 * specSample.disp0.cols, 0.4 * specSample.disp0.rows), 0, 0, cv::INTER_LINEAR);
		cv::resize(depth, depth, cv::Size(0.4 * depth.cols, 0.4 * depth.rows), 0, 0, cv::INTER_LINEAR);

		cv::imwrite(specSample.name + "_disp_ground.png", specSample.disp0);
		cv::imwrite(specSample.name + "_img.png", specSample.img1);

		cv::applyColorMap(specSample.disp0, specSample.disp0, cv::COLORMAP_JET);
		cv::imwrite(specSample.name + "_disp_ground_jet.png", specSample.disp0);

		depth = 255 * depth;
		cv::imwrite(specSample.name + "_depth_ground.png", depth);
	} else {
		std::cerr << "MAIN >> Nothing to do, check your MATCHING Marcos." << std::endl;
	}
#else
		std::cerr << "MAIN >> Nothing to do, check your SAMPLE Marcos.";
#endif
	return 0;
}


