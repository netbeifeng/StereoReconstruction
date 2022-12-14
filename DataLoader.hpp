#ifndef _DATALOADER_HPP
#define _DATALOADER_HPP

#include <iostream>
#include <fstream>
#include <random>
#include <filesystem>

#include "Utils.h"
#include "PFMManager.hpp"

/* DataLoader 
 * @brief for loading ImagePair and two camera intrinsic
 * as well as the groundtruth disparity map
 */
class DataLoader
{
    public:
        // Constructor for all loading
        DataLoader(std::string dataset) {
            std::cout << "DataLoader >> Loading data ..." << std::endl;
            m_dataset = dataset;
            std::filesystem::path rootDataPath = "../data/" + m_dataset;
            getFiles(rootDataPath, m_files, 100);
            initImagePairs();
            std::cout << "DataLoader >> All " << getSizeOfDataset() << " scenes loaded successfully." << std::endl;
        }
        // Constructor for loading specific number of imgs
        DataLoader(std::string dataset, int num) {
            std::cout << "DataLoader >> Loading data ..." << std::endl;
            m_dataset = dataset;
            std::filesystem::path rootDataPath = "../data/" + m_dataset;
            getFiles(rootDataPath, m_files, num);
            if (num > 0) {
                initImagePairs();
                std::cout << "DataLoader >> All " << getSizeOfDataset() << " scenes loaded successfully." << std::endl;
            }
            initImagePairs();
        }
        // Constructor for loading one specific image pair
        DataLoader(std::string dataset, std::string singleImgName) {
            std::cout << "DataLoader >> Loading for single image " << singleImgName << " ..." << std::endl;
            m_dataset = dataset;
            std::filesystem::path rootDataPath = "../data/" + m_dataset + "/" + singleImgName + "-perfect";
            getSpecificFiles(rootDataPath);
        }

        void initImagePairs() {
            for (int i = 0; i < m_files.size(); i++) {
                m_imagePairs.push_back(getImagePairByIndex(i));
            }
        }

        void getSpecificFiles(std::filesystem::path path) {
            m_files.push_back(path.string());
        }

        int getSizeOfDataset() {
            return m_files.size();
        }

        std::string getDataPathByIndex(int index) {
            return m_files.at(index);
        }

        ImagePair getSpecificSample() {
            return getImagePairByIndex(0);
        }

        ImagePair getImagePairByIndex(int index) {
            if (index < 0) {
                throw std::invalid_argument("Bad index");
            }

            std::string path = m_files.at(index);

            std::string img1Path = path + "/im0.png";
            std::string img2Path = path + "/im1.png";
            std::string calibPath = path + "/calib.txt";
            
            std::string disp0Path = path + "/disp0.pfm";
            std::string disp1Path = path + "/disp1.pfm";

            ImagePair imgPair;

            imgPair.path = path;

            imgPair.img1 = cv::imread(img1Path);
            imgPair.img2 = cv::imread(img2Path);

            cv::Mat disp0_raw = PFMManager::loadPFM(disp0Path);
            cv::Mat disp0_mask{disp0_raw == std::numeric_limits<float>::infinity()};
            disp0_raw.setTo(0, disp0_mask);

            cv::Mat disp0_norm;
            cv::normalize(disp0_raw, disp0_norm, 0, 255, cv::NORM_MINMAX);
            disp0_norm.convertTo(imgPair.disp0, CV_8UC1);

            cv::Mat disp1_raw = PFMManager::loadPFM(disp1Path);
            cv::Mat disp1_mask{disp1_raw == std::numeric_limits<float>::infinity()};
            disp1_raw.setTo(0, disp1_mask);

            cv::Mat disp1_norm;
            cv::normalize(disp1_raw, disp1_norm, 0, 255, cv::NORM_MINMAX);
            disp1_norm.convertTo(imgPair.disp1, CV_8UC1);

            std::ifstream ifs(calibPath, std::ios::in);

            if (!ifs.is_open()) {
                std::cout << "DataLoader >> Failed to open file.\n";
            } else {
                std::stringstream ss;
                ss << ifs.rdbuf();
                std::string str(ss.str());
                std::vector<std::string> lines = splitStringByNewline(str);
                imgPair.name = path.substr(path.find_last_of("/\\") + 1, path.length());
                imgPair.baseline = std::stof(getAttrNumByName("baseline", lines));
                imgPair.doffs = std::stof(getAttrNumByName("doffs", lines));
                imgPair.width = std::stoi(getAttrNumByName("width", lines));
                imgPair.height = std::stoi(getAttrNumByName("height", lines));
                imgPair.vmin = std::stoi(getAttrNumByName("vmin", lines));
                imgPair.vmax = std::stoi(getAttrNumByName("vmax", lines));

                imgPair.K_img1 = getIntrinsicOfString(getAttrNumByName("cam0", lines));
                imgPair.f1 = imgPair.K_img1.at<double>(0, 0);

                imgPair.K_img2 = getIntrinsicOfString(getAttrNumByName("cam1", lines));
                imgPair.f2 = imgPair.K_img2.at<double>(0, 0);

                ifs.close();
            }
            return imgPair;
        }

        std::vector<ImagePair> getAllImagePairs() {
            return m_imagePairs;
        }

        ImagePair getRandomSample() {
            std::vector<ImagePair> oneShot;
            std::sample(m_imagePairs.begin(), m_imagePairs.end(), std::back_inserter(oneShot),
                1, std::mt19937{ std::random_device{}() });
            return oneShot.at(0);
        }

        cv::Mat getIntrinsicOfString(std::string intrinsicString) {
            auto result = std::vector<std::string>{};
            auto ss = std::stringstream{ intrinsicString };

            for (std::string line; std::getline(ss, line, ';');)
                result.push_back(line);

            auto row = std::vector<double>{};

            for (std::string line : result) {
                auto ss_sub = std::stringstream{ line };
                for (std::string line_sub; std::getline(ss_sub, line_sub, ' ');) {
                    line_sub.erase(std::remove(line_sub.begin(), line_sub.end(), ' '), line_sub.end());
                    if (line_sub.length() > 0) {
                        row.push_back(std::stof(line_sub));
                    }
                }
            }
            
            cv::Mat retMat = (cv::Mat_<double>(3, 3) << row[0], row[1], row[2],
                                                        row[3], row[4], row[5],
                                                        row[6], row[7], row[8]);
            return retMat;
        }

    private:
        std::string m_dataset = "Middlebury_2014";
        std::vector<ImagePair> m_imagePairs;
        std::vector<std::string> m_files;
        
        void getFiles(std::filesystem::path path, std::vector<std::string>& m_files, int num, bool specific = false)
        {
            for (const auto& entry : std::filesystem::directory_iterator(path)) {
                m_files.push_back(entry.path().string());
            }
            if (num < 0) {
                return;
            }

            if (num < m_files.size()) {
                std::vector<int> tmpIndcies, outIndices;
                for (int i = 0; i < m_files.size(); i++) {
                    tmpIndcies.push_back(i);
                }

                std::sample(tmpIndcies.begin(), tmpIndcies.end(), std::back_inserter(outIndices),
                    num, std::mt19937{ std::random_device{}() });

                std::vector<std::string> tmpFiles;
                for (int i = 0; i < outIndices.size(); i++) {
                    std::cout << "DataLoader >> " << m_files.at(outIndices.at(i)) << " is loaded." << std::endl;
                    tmpFiles.push_back(m_files.at(outIndices.at(i)));
                }

                m_files.clear();
                m_files.swap(tmpFiles);
            }

            else {
                for (int i = 0; i < m_files.size(); i++) {
                    std::cout << "DataLoader >> " << m_files.at(i) << " is loaded." << std::endl;
                }
            }
        }

        std::string getAttrNumByName(std::string attr, std::vector<std::string> lines) {
            int offset = 1;
            int backOffset = 0;
            for (std::string line : lines) {
                if (line.find(attr) == 0) {
                    if (attr.find("cam") == 0) {
                        offset++;
                        backOffset--;
                    }
                    return line.substr(attr.length() + offset, line.length() - attr.length() - offset + backOffset);
                }
            }

            return "ERROR";
        }

        std::vector<std::string> splitStringByNewline(const std::string& str)
        {
            auto result = std::vector<std::string>{};
            auto ss = std::stringstream{ str };

            for (std::string line; std::getline(ss, line, '\n');)
                result.push_back(line);

            return result;
        }
};
#endif