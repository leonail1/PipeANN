cmake_minimum_required(VERSION 3.16)
project(openharmony_anns_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(nlohmann_json REQUIRED)

find_library(FAISS_LIBRARY faiss)
if(NOT FAISS_LIBRARY)
  message(FATAL_ERROR "FAISS library not found")
endif()

add_executable(openharmony_anns_adapter
  src/openharmony_anns_adapter.cpp
)

target_compile_options(openharmony_anns_adapter PRIVATE -O3 -Wall -Wextra -Wpedantic)

target_link_libraries(openharmony_anns_adapter PRIVATE
  ${FAISS_LIBRARY}
  OpenMP::OpenMP_CXX
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
  nlohmann_json::nlohmann_json
)
