cmake_minimum_required(VERSION 3.16)
project(hybrid_rebuild_context LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(HYBRID_CONTEXT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(PIPEANN_ROOT "${HYBRID_CONTEXT_ROOT}/..")
set(GENERATED_INCLUDE_DIR "${CMAKE_BINARY_DIR}/generated_include")
set(HYBRID_RUNTIME_DIR "${CMAKE_BINARY_DIR}/bin")

file(MAKE_DIRECTORY "${GENERATED_INCLUDE_DIR}/nbr")
configure_file("${HYBRID_CONTEXT_ROOT}/cmake/nbr.h.in" "${GENERATED_INCLUDE_DIR}/nbr/nbr.h" @ONLY)

file(READ "${PIPEANN_ROOT}/include/ssd_index.h" SSD_INDEX_HEADER_CONTENT)
string(
  REPLACE
  "    int get_vector_by_id(const uint32_t &id, T *vector);\n\n    // ID, loc, page mapping.\n    TagT id2tag(uint32_t id);\n"
  "   public:\n    int get_vector_by_id(const uint32_t &id, T *vector);\n\n    // ID, loc, page mapping.\n    TagT id2tag(uint32_t id);\n\n   private:\n"
  SSD_INDEX_HEADER_CONTENT
  "${SSD_INDEX_HEADER_CONTENT}"
)
file(WRITE "${GENERATED_INCLUDE_DIR}/ssd_index.h" "${SSD_INDEX_HEADER_CONTENT}")

find_package(OpenMP REQUIRED)
find_library(TCMALLOC_LIB NAMES tcmalloc tcmalloc_minimal REQUIRED)

set(URING_LIB "${PIPEANN_ROOT}/third_party/liburing/src/liburing.a")
if(NOT EXISTS "${URING_LIB}")
  message(FATAL_ERROR "Missing liburing static library at ${URING_LIB}")
endif()

add_library(
  pipeann_hybrid
  STATIC
  "${PIPEANN_ROOT}/src/index.cpp"
  "${PIPEANN_ROOT}/src/pipnn.cpp"
  "${PIPEANN_ROOT}/src/ssd_index.cpp"
  "${PIPEANN_ROOT}/src/search/beam_search.cpp"
  "${PIPEANN_ROOT}/src/search/pipe_search.cpp"
  "${PIPEANN_ROOT}/src/update/direct_insert.cpp"
  "${PIPEANN_ROOT}/src/utils/linux_aligned_file_reader.cpp"
  "${PIPEANN_ROOT}/src/utils/distance.cpp"
  "${PIPEANN_ROOT}/src/utils/index_build_utils.cpp"
  "${PIPEANN_ROOT}/src/utils/kmeans_utils.cpp"
  "${PIPEANN_ROOT}/src/utils/partition.cpp"
)

target_include_directories(
  pipeann_hybrid
  PUBLIC
    "${GENERATED_INCLUDE_DIR}"
    "${HYBRID_CONTEXT_ROOT}/include"
    "${PIPEANN_ROOT}/include"
    "${PIPEANN_ROOT}/include/tsl/include"
    "${PIPEANN_ROOT}/third_party/liburing/src/include"
)
target_link_libraries(pipeann_hybrid PUBLIC OpenMP::OpenMP_CXX PRIVATE "${URING_LIB}" "${TCMALLOC_LIB}")
target_compile_definitions(pipeann_hybrid PRIVATE DNDEBUG DBG_IO_THREAD OVERLAP_INIT)
target_compile_options(
  pipeann_hybrid
  PRIVATE
    -fopenmp-simd
    -Wall
    -Wextra
    -Wfatal-errors
    -Wno-class-memaccess
    -Wno-unused-variable
    -Wno-unused-parameter
    -Wno-unused-but-set-variable
    -g
    -O3
    -march=native
    -mtune=native
    -ftree-vectorize
    -funroll-loops
    -Wl,--no-as-needed
)

function(add_hybrid_binary target source_file)
  add_executable("${target}" "${source_file}")
  target_link_libraries("${target}" PRIVATE pipeann_hybrid)
  set_target_properties("${target}" PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${HYBRID_RUNTIME_DIR}")
endfunction()

add_hybrid_binary(hybrid_build_disk_index "${PIPEANN_ROOT}/tests/build_disk_index.cpp")
add_hybrid_binary(hybrid_search_disk_index_filtered "${HYBRID_CONTEXT_ROOT}/src/search_disk_index_filtered.cpp")
add_hybrid_binary(
  hybrid_search_disk_index_filtered_prefilter
  "${HYBRID_CONTEXT_ROOT}/src/search_disk_index_filtered_prefilter.cpp"
)

add_custom_target(
  hybrid_tools
  DEPENDS
    hybrid_build_disk_index
    hybrid_search_disk_index_filtered
    hybrid_search_disk_index_filtered_prefilter
)
