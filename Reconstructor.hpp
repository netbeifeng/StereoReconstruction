#ifndef _RECONSTRUCTOR_HPP
#define _RECONSTRUCTOR_HPP
#include <iostream>
#include <fstream>
#include <array>
#include <vector>
#include <opencv2/core/core.hpp>

#include "Eigen.h"
#include <opencv2/core/eigen.hpp>

#include "Utils.h"

// Print debug information when set to 1
#define DEBUG 1

#define EPSILON 10e-5
/***
 *  Reconstruction with .off format mesh files
 *  Reference 3D Scanning and Motion capture SS21 TUM Lecture materials
 * ***/
struct Vertex
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // position stored as 4 floats (4th component is supposed to be 1.0)
    Vector4f position;
    // color stored as 4 unsigned char
    Vector4uc color;
};

class Mesh
{
public:
    static bool writeMesh(Vertex *vertices, int width, int height, const std::string &filename)
    {
        float edgeThreshold = 0.01f; // 1cm
        unsigned int nVertices = width * height;

        // Maximum number of faces
        // One is missing (rightmost column, bottom row)
        // and for each vertex, there are 2 triangles
        // 		=> hence: (width - 1) * (height - 1) * 2
        unsigned nFaces = (width - 1) * (height - 1) * 2;

        std::vector<Eigen::Vector3i> faces{nFaces};
        int faceCount = 0;

        int idx = 0;
        for (int x = 0; x < (width - 1); x++)
        {
            for (int y = 0; y < (height - 1); y++, idx++)
            {
                int current = idx;
                int bottom = idx + width;
                int right = idx + 1;
                int diag = idx + width + 1;

                // Left upper face
                if (Mesh::valid_triangle(vertices, current, bottom, right, edgeThreshold))
                {
                    faces[faceCount++] = Vector3i{current, bottom, right};
                }

                // Right lower face
                if (Mesh::valid_triangle(vertices, bottom, diag, right, edgeThreshold))
                {
                    faces[faceCount++] = Vector3i{bottom, diag, right};
                }
            }
        }

        nFaces = faceCount;

        // Write off file
        std::ofstream outFile(filename);
        if (!outFile.is_open())
            return false;

        // write header
        outFile << "COFF" << std::endl;
        outFile << "# numVertices numFaces numEdges" << std::endl;
        outFile << nVertices << " " << nFaces << " 0" << std::endl;

        for (int i = 0; i < width * height; i++)
        {
            if (vertices[i].position[0] == MINF || vertices[i].position[1] == MINF || vertices[i].position[2] == MINF)
            {
                outFile << "0.0 0.0 0.0 0 0 0\n";
            }
            else {
                outFile << vertices[i].position[0] << " "
                    << vertices[i].position[1] << " "
                    << vertices[i].position[2] << " "
                    << (int)vertices[i].color[0] << " "
                    << (int)vertices[i].color[1] << " "
                    << (int)vertices[i].color[2] << "\n";
            }
        }

        for (int i = 0; i < nFaces; i++)
        {
            outFile << "3 "
                    << faces[i][0] << " "
                    << faces[i][1] << " "
                    << faces[i][2] << "\n";
        }

        outFile.close();
        return true;
    }

private:
    // Checks for the given vertices, if the triangle formed by them is valid.
    // This is only the case if every edge is smaller than some threshold value.
    static bool valid_triangle(Vertex *vertices, int i0, int i1, int i2, float edgeThreshold)
    {
        return (vertices[i0].position - vertices[i1].position).norm() < edgeThreshold && (vertices[i0].position - vertices[i2].position).norm() < edgeThreshold && (vertices[i1].position - vertices[i2].position).norm() < edgeThreshold;
    }
};

class Reconstructor
{

public:
    // Compute a depth from from the given disperity map in `disperity` using the
    // intrinsic parameters provided by the `intrisicMat` matrix. If `normalize` is
    // true, the computed depth values are scaled to the interval [0-1].
    // The computed depth map is of type CV_32FC1, this a 1-channel 32-bit floating point matrix.
    cv::Mat depthMapFromDisperityMap(cv::Mat disperity, float baseline, float doffs, float focal, float *maxDepth, bool normalize = false)
    {
        cv::Mat depth_map(disperity.size(), CV_32FC1);

        int rows = depth_map.rows;
        int cols = depth_map.cols;

        for (int x = 0; x < cols; x++)
        {
            for (int y = 0; y < rows; y++)
            {
                float d = disperity.at<unsigned char>(y, x);
                float depth_val = baseline * focal / (d + doffs);
                depth_map.at<float>(y, x) = depth_val;
            }
        }
        std::cout << 3 << std::endl;

        double min, max;
        cv::minMaxLoc(depth_map, &min, &max);

        *maxDepth = (float)max;

#ifdef DEBUG
        std::cout << "Reconstruction >> Depth map minimum value " << min << "\n";
        std::cout << "Reconstruction >> Depth map maximum value " << max << "\n";
#endif

        if (normalize)
        {
            cv::Mat depth_map_norm;
            cv::normalize(depth_map, depth_map_norm, 0.0, 1.0, cv::NORM_MINMAX);
            *maxDepth = 1.0;
            return depth_map_norm;
        }

        return depth_map;
    }

    cv::Mat depthMapFromNormDisperity(cv::Mat disperity, float baseline, float doffs, float focal, float *maxDepth, bool normalize = false)
    {
        cv::Mat depth_map(disperity.size(), CV_32FC1);

        int rows = depth_map.rows;
        int cols = depth_map.cols;

        for (int x = 0; x < cols; x++)
        {
            for (int y = 0; y < rows; y++)
            {
                float d = disperity.at<unsigned char>(y, x);
                float depth_val = baseline * focal / (d + doffs);
                depth_map.at<float>(y, x) = depth_val;
            }
        }

        double min, max;
        cv::minMaxLoc(depth_map, &min, &max);

        *maxDepth = (float)max;

#ifdef DEBUG
        std::cout << "Reconstruction >> Depth map minimum value " << min << "\n";
        std::cout << "Reconstruction >> Depth map maximum value " << max << "\n";
#endif

        if (normalize)
        {
            cv::Mat depth_map_norm;
            cv::normalize(depth_map, depth_map_norm, 0.0, 1.0, cv::NORM_MINMAX);
            *maxDepth = 1.0;
            return depth_map_norm;
        }

        return depth_map;
    }

    // Generate a point cloud given a depth map in `depthMap` with the colors provided by `colorMap`.
    // Returns a pointer to an array of vertices which is the same size as the color/depth image.
    // The array of vertices needs to be deallocated after use.
    // `maxDepth` is the maximum depth value that is attained in the depth map.
    Vertex *generatePointCloud(cv::Mat depthMap, cv::Mat colorMap, cv::Mat depthIntrinsicMat, float maxDepth = 1.0)
    {
        int width = depthMap.cols;
        int height = depthMap.rows;

        Eigen::Matrix3f depthIntrinsic;
        cv::cv2eigen(depthIntrinsicMat, depthIntrinsic);

        Eigen::Matrix3f depthIntrinsicInv = depthIntrinsic.inverse();
        Vertex *vertices = new Vertex[width * height];

        int idx = 0;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++, idx++)
            {
                float depth = depthMap.at<float>(y, x);

                if (std::abs(depth - maxDepth) < EPSILON)
                {
                    vertices[idx].position = Eigen::Vector4f(MINF, MINF, MINF, MINF);
                    vertices[idx].color = Vector4uc(0, 0, 0, 0);
                }
                else
                {
                    Eigen::Vector3f img_coords = Eigen::Vector3f(x * depth, y * depth, depth);
                    Eigen::Vector3f tmp_coords = depthIntrinsicInv * img_coords;

                    Eigen::Vector4f world_coords = Eigen::Vector4f(tmp_coords[0], tmp_coords[1], tmp_coords[2], 1.0f);

                    // Format is B,G,R
                    cv::Vec3b color = colorMap.at<cv::Vec3b>(y, x);

                    vertices[idx].position = world_coords;
                    vertices[idx].color = Vector4uc(color[2], color[1], color[0], 255);
                }
            }
        }

        return vertices;
    }
};
#endif