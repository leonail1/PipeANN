find_package(OpenMP REQUIRED)

set(OH_ACCEPTANCE_COMMON_INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

function(oh_add_acceptance_exe target source)
  add_executable(${target} ${source})
  target_include_directories(${target} PRIVATE ${OH_ACCEPTANCE_COMMON_INCLUDE})
  target_link_libraries(${target} PRIVATE pipeann OpenMP::OpenMP_CXX)
  target_compile_features(${target} PRIVATE cxx_std_17)
endfunction()

oh_add_acceptance_exe(oh_generate_labels src/oh_generate_labels.cpp)
oh_add_acceptance_exe(oh_make_synthetic src/oh_make_synthetic.cpp)
oh_add_acceptance_exe(oh_materialize_cycle_vectors src/oh_materialize_cycle_vectors.cpp)
oh_add_acceptance_exe(oh_static_filtered src/oh_static_filtered.cpp)
oh_add_acceptance_exe(oh_dynamic_chain src/oh_dynamic_chain.cpp)
oh_add_acceptance_exe(oh_single_query src/oh_single_query.cpp)
oh_add_acceptance_exe(oh_rebuild_pq src/oh_rebuild_pq.cpp)
oh_add_acceptance_exe(oh_summarize_results src/oh_summarize_results.cpp)

add_executable(oh_build_space src/oh_build_space.cpp)
target_include_directories(oh_build_space PRIVATE ${OH_ACCEPTANCE_COMMON_INCLUDE})
target_compile_features(oh_build_space PRIVATE cxx_std_17)
