cmake_minimum_required(VERSION 3.16)
project(cvi_minimal_runner C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

if(NOT DEFINED TPU_SDK_PATH)
  get_filename_component(TPU_SDK_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../duo-tpu/workspace/tpu-sdk" ABSOLUTE)
endif()

message(STATUS "TPU_SDK_PATH=${TPU_SDK_PATH}")

add_executable(cvi_minimal_runner src/main.cpp)

target_include_directories(cvi_minimal_runner PRIVATE
  "${TPU_SDK_PATH}/include"
)

target_link_directories(cvi_minimal_runner PRIVATE
  "${TPU_SDK_PATH}/lib"
)

target_link_libraries(cvi_minimal_runner PRIVATE
  cviruntime
  cvikernel
  cnpy
  z
  dl
)

set_target_properties(cvi_minimal_runner PROPERTIES
  BUILD_RPATH "/mnt/tpu/tpu-sdk/lib"
  INSTALL_RPATH "/mnt/tpu/tpu-sdk/lib"
)
