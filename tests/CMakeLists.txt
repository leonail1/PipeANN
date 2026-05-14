set(CMAKE_CXX_STANDARD 17)

add_executable(build_disk_index build_disk_index.cpp)
target_link_libraries(build_disk_index ${PROJECT_NAME})

add_executable(build_pipnn_index build_pipnn_index.cpp)
target_link_libraries(build_pipnn_index ${PROJECT_NAME})

add_executable(build_memory_index build_memory_index.cpp)
target_link_libraries(build_memory_index ${PROJECT_NAME})

add_executable(calibrate_hybrid_threshold calibrate_hybrid_threshold.cpp)
target_link_libraries(calibrate_hybrid_threshold ${PROJECT_NAME})

add_executable(hybrid_metadata_roundtrip hybrid_metadata_roundtrip.cpp)
target_link_libraries(hybrid_metadata_roundtrip ${PROJECT_NAME})

add_executable(densebit_range densebit_range.cpp)
target_link_libraries(densebit_range ${PROJECT_NAME})

add_executable(overall_performance overall_performance.cpp)
target_link_libraries(overall_performance ${PROJECT_NAME})

add_executable(test_insert_search test_insert_search.cpp)
target_link_libraries(test_insert_search ${PROJECT_NAME})

add_executable(dynamic_prefilter_stage_driver dynamic_prefilter_stage_driver.cpp)
target_link_libraries(dynamic_prefilter_stage_driver ${PROJECT_NAME})

add_executable(dynamic_update_suite_driver dynamic_update_suite_driver.cpp)
target_link_libraries(dynamic_update_suite_driver ${PROJECT_NAME})

add_executable(dynamic_flat_index dynamic_flat_index.cpp)
target_link_libraries(dynamic_flat_index ${PROJECT_NAME})

add_executable(dynamic_flat_sift1m_smoke dynamic_flat_sift1m_smoke.cpp)
target_link_libraries(dynamic_flat_sift1m_smoke ${PROJECT_NAME})

add_executable(dynamic_flat_small_search_bench dynamic_flat_small_search_bench.cpp)
target_link_libraries(dynamic_flat_small_search_bench ${PROJECT_NAME})

add_executable(overall_perf_mem overall_perf_mem.cpp)
target_link_libraries(overall_perf_mem ${PROJECT_NAME})

add_executable(search_disk_index search_disk_index.cpp)
target_link_libraries(search_disk_index ${PROJECT_NAME})

add_executable(search_disk_index_mem search_disk_index_mem.cpp)
target_link_libraries(search_disk_index_mem ${PROJECT_NAME})

add_executable(search_disk_index_filtered search_disk_index_filtered.cpp)
target_link_libraries(search_disk_index_filtered ${PROJECT_NAME})

add_executable(search_disk_index_hybrid search_disk_index_hybrid.cpp)
target_link_libraries(search_disk_index_hybrid ${PROJECT_NAME})
add_dependencies(build_disk_index search_disk_index_hybrid)

add_executable(pad_partition pad_partition.cpp)
target_link_libraries(pad_partition ${PROJECT_NAME} )

add_executable(normalize_data normalize_data.cpp)
target_link_libraries(normalize_data ${PROJECT_NAME})
