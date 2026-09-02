# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES "87")

project(matx_reformat LANGUAGES CUDA CXX)
find_package(CUDAToolkit 11.4 REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 87)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Number of classes of the deployed model — must match the app build's
# NUM_CLASSES (build with: NUM_CLASSES=<n> bash build_matx_reformat.sh)
set(YOLO_NUM_CLASSES "80" CACHE STRING "classes of the deployed YOLOv5 model")
add_compile_definitions(YOLO_NUM_CLASSES=${YOLO_NUM_CLASSES})
set(YOLO_INPUT_W "672" CACHE STRING "network input width, multiple of 32")
set(YOLO_INPUT_H "672" CACHE STRING "network input height, multiple of 32")
add_compile_definitions(YOLO_INPUT_W=${YOLO_INPUT_W} YOLO_INPUT_H=${YOLO_INPUT_H})
# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

# --- CUDA 12.x (JetPack 6) compatibility -----------------------------------
# MatX 0.4.1 pins libcudacxx 2.1.0, but nvcc always adds CUDA 12.6's newer CCCL
# headers too; mixing the two generations breaks the build (NVTX v1/v3 #error,
# missing <__config>). Force MatX to consume the toolkit's own CCCL by pointing
# its libcudacxx fetch at a shim whose include/ is the toolkit include dir, and
# put the NVTX v1 -> v3 redirect first on the include path.
set(FETCHCONTENT_SOURCE_DIR_LIBCUDACXX "${CMAKE_CURRENT_SOURCE_DIR}/compat/libcudacxx-shim")
include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/compat/nvtx-shim")

# Linker flags
add_subdirectory(MatX)

# Source files
set(SRCS matx_reformat.cu)

# Library name
set(LIBNAME matx_reformat)

add_library(${LIBNAME} SHARED ${SRCS})
target_link_libraries(${LIBNAME} ${CUDA_LIBRARIES})
target_link_libraries(${LIBNAME} matx::matx)

add_executable(test test.cpp)
target_include_directories(test PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(test ${LIBNAME} ${CUDA_LIBRARIES})
target_link_libraries(test matx::matx)