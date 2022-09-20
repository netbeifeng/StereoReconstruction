cmake_minimum_required(VERSION 3.6)

SET(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)
SET(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE PATH "Path to your vcpkg")

PROJECT(stereo_reconstruction_ss22)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)

FIND_PACKAGE(Eigen3 CONFIG REQUIRED)
FIND_PACKAGE(OpenCV CONFIG REQUIRED)
find_package(suitesparse CONFIG REQUIRED)
FIND_PACKAGE(g2o CONFIG REQUIRED)

SET(HEADERS
    FeatureDectector.h
    BAOptimizer.h
    BlockMatcher.h
    Utils.h
    Eigen.h
    EightPoint.h
    Reconstructor.hpp
    DataLoader.hpp
    PFMManager.hpp
)

SET(SOURCES
    FeatureDectector.cpp
    EightPoint.cpp
    BlockMatcher.cpp
    BAOptimizer.cpp
    main.cpp
)

add_executable(stereo_reconstruction_ss22 ${HEADERS} ${SOURCES})

LINK_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

target_include_directories(stereo_reconstruction_ss22 PUBLIC ${OpenCV_INCLUDE_DIRS})
target_include_directories(stereo_reconstruction_ss22 PUBLIC ${EIGEN3_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(stereo_reconstruction_ss22 ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(stereo_reconstruction_ss22 Eigen3::Eigen)
TARGET_LINK_LIBRARIES(stereo_reconstruction_ss22 SuiteSparse::amd SuiteSparse::btf SuiteSparse::klu SuiteSparse::ldl)
TARGET_LINK_LIBRARIES(stereo_reconstruction_ss22 g2o::core g2o::stuff g2o::types_icp g2o::types_sba)
