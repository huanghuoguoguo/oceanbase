
add_subdirectory (simd)

file (GLOB CPP_SRCS "*.cpp")
file (GLOB CPP_FACTORY_SRCS "factory/*.cpp")
file (GLOB CPP_CONJUGATE_GRAPH_SRCS "impl/*.cpp")
file (GLOB CPP_INDEX_SRCS "index/*.cpp")
file (GLOB CPP_HNSWLIB_SRCS "algorithm/hnswlib/*.cpp")
list (FILTER CPP_SRCS EXCLUDE REGEX "_test.cpp")
list (FILTER CPP_FACTORY_SRCS EXCLUDE REGEX "_test.cpp")
list (FILTER CPP_CONJUGATE_GRAPH_SRCS EXCLUDE REGEX "_test.cpp")
list (FILTER CPP_INDEX_SRCS EXCLUDE REGEX "_test.cpp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set (VSAG_SRCS ${CPP_SRCS} ${CPP_FACTORY_SRCS} ${CPP_INDEX_SRCS} ${CPP_CONJUGATE_GRAPH_SRCS} ${CPP_HNSWLIB_SRCS})
add_library (vsag_test SHARED ${VSAG_SRCS})
add_library (vsag_static_test STATIC ${VSAG_SRCS})
#target_link_directories(
#  vsag_static_test PUBLIC 
#  ${CMAKE_BINARY_DIR}/deps/oblib/src/lib/vector/vsag_lib/src/simd
#  ${DEP_DIR}/lib ${DEP_DIR}/lib/vsag_lib
#  ${CMAKE_BINARY_DIR}/deps/oblib/src/lib/vector/vsag_lib 
#  ${CMAKE_BINARY_DIR}/deps/oblib/src/lib/vector/vsag_lib/src
#  ${CMAKE_BINARY_DIR}/_deps/fmt-build
#)
# 指定libgomp_static.a的路径
# link_directories(/root/source/oceanbase/deps/3rd/usr/local/oceanbase/deps/devel/lib/vsag_lib)

# 链接libgomp_static.a
# target_link_libraries(vsag_test  "/root/source/oceanbase/deps/3rd/usr/local/oceanbase/deps/devel/lib/vsag_lib/libgomp_static.a")
# target_link_libraries(vsag_static_test  "/root/source/oceanbase/deps/3rd/usr/local/oceanbase/deps/devel/lib/vsag_lib/libgomp_static.a")

set (VSAG_DEP_LIBS diskann pthread m dl simd fmt::fmt-header-only nlohmann_json::nlohmann_json roaring)

# 链接依赖库
target_link_libraries (vsag_test  ${VSAG_DEP_LIBS})
target_link_libraries (vsag_static_test  ${VSAG_DEP_LIBS})
#target_link_libraries (vsag_static_test  libgomp_static.a)
#target_link_libraries (vsag_static_test  omp)


maybe_add_dependencies (vsag_test spdlog roaring openblas boost mkl)
maybe_add_dependencies (vsag_static_test spdlog roaring openblas boost mkl)


