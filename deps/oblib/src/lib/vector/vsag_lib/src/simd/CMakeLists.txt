set (SIMD_SRCS generic.cpp simd.cpp) 

if (DIST_CONTAINS_SSE) 
  set (SIMD_SRCS ${SIMD_SRCS} sse.cpp) 
  set_source_files_properties (sse.cpp PROPERTIES COMPILE_FLAGS "-msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -msse4.2") 
  add_definitions (-DENABLE_SSE=1) 
endif () 

if (DIST_CONTAINS_AVX) 
  set (SIMD_SRCS ${SIMD_SRCS} avx.cpp) 
  set_source_files_properties (avx.cpp PROPERTIES COMPILE_FLAGS "-mavx") 
  add_definitions (-DENABLE_AVX=1) 
endif () 

if (DIST_CONTAINS_AVX2) 
  set_source_files_properties (avx.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma") 
  add_definitions (-DENABLE_AVX2=1) 
endif () 

if (DIST_CONTAINS_AVX512) 
  set (SIMD_SRCS ${SIMD_SRCS} avx512.cpp) 
  set_source_files_properties (avx512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512pf -mavx512er -mavx512cd -mavx512vl -mavx512bw -mavx512dq -mavx512ifma -mavx512vbmi") 
  add_definitions (-DENABLE_AVX512=1) 
endif () 

# OpenMP setup
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fopenmp-simd")
  target_link_libraries(simd PRIVATE OpenMP::OpenMP_CXX)
endif()

# Additional compile flags
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-calloc")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-realloc")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-free")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")

# Create library
add_library (simd ${SIMD_SRCS}) 

# Link dependencies
target_link_libraries (simd PRIVATE cpuinfo) 

# Install
install (TARGETS simd ARCHIVE DESTINATION lib)
