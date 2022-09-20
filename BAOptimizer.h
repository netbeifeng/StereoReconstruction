#ifndef _BAOPTIMIZER_H
#define _BAOPTIMIZER_H

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include "Utils.h"

class BAOptimizer{
    public: 
        BAOptimizer(ImagePair pair, std::pair<KeyPoints, KeyPoints> cpoints);
        std::pair<Rotate, Translate> optimize(std::pair<Rotate, Translate> pose, int iterNum);

    private:
        ImagePair m_pair;
        g2o::SparseOptimizer m_optimizer;
        std::pair<KeyPoints, KeyPoints> m_cpoints;
};
#endif