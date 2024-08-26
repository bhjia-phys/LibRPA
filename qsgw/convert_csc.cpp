#include <iostream>
#include <filesystem>
#include <regex>
#include "utils.h"

namespace fs = std::filesystem;
using namespace Eigen;
using namespace std;

void process_csc(const std::string& filePath, std::map<std::string, MatrixXcd>& matrices) {
    std::regex filePattern(R"((\w+)_spin_(\d+)_kpt_(\d+)(?:_freq_(\d+))?.csc)");
    std::smatch match;

    if (std::regex_search(filePath, match, filePattern)) {
        std::string fileType = match[1];
        int ispin = std::stoi(match[2]) - 1;  // 将自旋从1基索引转换为0基索引
        int ikpt = std::stoi(match[3]) - 1;   // 将k点从1基索引转换为0基索引
        int freq = match[4].matched ? std::stoi(match[4]) : -1;

        std::cout << "File Type: " << fileType << std::endl;
        std::cout << "Spin: " << ispin << std::endl;
        std::cout << "K-point: " << ikpt << std::endl;
        if (freq != -1) {
            std::cout << "Frequency: " << freq << std::endl;
        }

        std::cout << "尝试打开文件: " << filePath << std::endl;

        MatrixXcd matrix;
        loadMatrix(filePath, matrix);

        // 生成唯一键来存储矩阵数据，例如： "vxc_spin_0_kpt_0_freq_-1"
        std::string key = fileType + "_spin_" + std::to_string(ispin) + "_kpt_" + std::to_string(ikpt);
        if (freq != -1) {
            key += "_freq_" + std::to_string(freq);
        }

        // 将矩阵存储在 map 中，使用文件类型和索引作为键
        matrices[key] = matrix;

        std::cout << "Matrix loaded and stored successfully under key: " << key << std::endl;
    } else {
        std::cerr << "无法解析文件名: " << filePath << std::endl;
    }
}
