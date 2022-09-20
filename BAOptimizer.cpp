#include "BAOptimizer.h"

BAOptimizer::BAOptimizer(ImagePair pair, std::pair<KeyPoints, KeyPoints> cpoints) {
    m_pair = pair;
    m_cpoints = cpoints;

	typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
	typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;

	// Initialization of Optimizer
	auto solver = new g2o::OptimizationAlgorithmLevenberg(
		g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

	m_optimizer.setAlgorithm(solver);
	m_optimizer.setVerbose(true);

    std::cout << "BAOptimizer >> Initialization done." << std::endl;
	double baseline = m_pair.baseline;
}

std::pair<Rotate, Translate> BAOptimizer::optimize(std::pair<Rotate, Translate> RT, int iterNum) {
    Rotate R = RT.first;
    Translate T = RT.second;
	// Take the first pose backprojection as 3D point
	for (int i = 0; i < 2; i++)
	{
		g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap();
		v->setId(i);
		if (i == 0)
			v->setFixed(true); // fix the first point

		if (i == 0) {
			v->setEstimate(g2o::SE3Quat());
		}
		else {
			g2o::Vector3 _t;
			_t << T.at<double>(0, 0), T.at<double>(1, 0), T.at<double>(2, 0);
			g2o::Matrix3 _R;
			_R << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
				R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
				R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
			v->setEstimate(g2o::SE3Quat(_R, _t));
		}

		m_optimizer.addVertex(v);
	}

    for (int i = 0; i < m_cpoints.first.size(); i++)
	{
		g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
		v->setId(2 + i);
		double z = 1.0;
		double x = (m_cpoints.first[i].x - m_pair.K_img1.at<double>(0, 2)) * z / m_pair.K_img1.at<double>(0, 0);
		double y = (m_cpoints.first[i].y - m_pair.K_img1.at<double>(1, 2)) * z / m_pair.K_img1.at<double>(0, 0);
		v->setMarginalized(true);
		v->setEstimate(Eigen::Vector3d(x, y, z));
		m_optimizer.addVertex(v);
	}

	// Setup two cameras
    g2o::CameraParameters* camera = new g2o::CameraParameters(
        m_pair.K_img1.at<double>(0, 0), 
        Eigen::Vector2d(m_pair.K_img1.at<double>(0, 2), m_pair.K_img1.at<double>(1, 2)), 0);
	camera->setId(0);
	m_optimizer.addParameter(camera);

	g2o::CameraParameters* camera2 = new g2o::CameraParameters(
        m_pair.K_img2.at<double>(0, 0), 
        Eigen::Vector2d(m_pair.K_img2.at<double>(0, 2), m_pair.K_img2.at<double>(1, 2)), 0);
	camera2->setId(1);
	m_optimizer.addParameter(camera2);

    std::vector<g2o::EdgeProjectXYZ2UV*> edges;
	
	// fill edges
	for (int i = 0; i < m_cpoints.first.size(); i++)
	{
		g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
		edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*>   (m_optimizer.vertex(i + 2)));
		edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap*>     (m_optimizer.vertex(0)));
		edge->setMeasurement(Eigen::Vector2d(m_cpoints.first[i].x, m_cpoints.first[i].y));
		edge->setInformation(Eigen::Matrix2d::Identity());
		edge->setParameterId(0, 0);
		edge->setRobustKernel(new g2o::RobustKernelHuber());
		m_optimizer.addEdge(edge);
		edges.push_back(edge);
	}

	for (int i = 0; i < m_cpoints.second.size(); i++)
	{
		g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
		edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*>   (m_optimizer.vertex(i + 2)));
		edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap*>     (m_optimizer.vertex(1)));
		edge->setMeasurement(Eigen::Vector2d(m_cpoints.second[i].x, m_cpoints.second[i].y));
		edge->setInformation(Eigen::Matrix2d::Identity());
		edge->setParameterId(0, 1);
		edge->setRobustKernel(new g2o::RobustKernelHuber());
		m_optimizer.addEdge(edge);
		edges.push_back(edge);
	}

    std::cout << "BAOptimizer >> Start Optimizing..." << std::endl;
	if (!OPTIMIZATION_LOG_VERBOSE) {
		m_optimizer.setVerbose(false);
	}

	m_optimizer.initializeOptimization();
	m_optimizer.optimize(iterNum);
	std::cout << "BAOptimizer >> Optimization done." << std::endl;

	// Retriving the optimized pose
	g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>(m_optimizer.vertex(1));
	Eigen::Isometry3d pose = v->estimate();
	Eigen::Vector3d translation = pose.translation() / pose.translation().norm() * m_pair.baseline * 0.001;

    Translate newT = (cv::Mat_<double>(3, 1) << translation[0], translation[1], translation[2]);
    Rotate newR = (cv::Mat_<double>(3, 3) << pose.rotation()(0, 0), pose.rotation()(0, 1), pose.rotation()(0, 2),
											pose.rotation()(1, 0), pose.rotation()(1, 1), pose.rotation()(1, 2),
											pose.rotation()(2, 0), pose.rotation()(2, 1), pose.rotation()(2, 2));
    return std::make_pair(newR, newT);
}