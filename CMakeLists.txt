cmake_minimum_required(VERSION 3.14)
project(trt_plugin LANGUAGES CXX CUDA)

SET(CXXFLAGS
        -pipe
        -W
        -Wall
        -Wextra
        -m64
        -Wno-invalid-offsetof
        -Wno-deprecated
        -Wno-deprecated-declarations
        -Wno-unused-parameter
        -Wno-sign-compare
        -Wno-write-strings
        -Wno-unused-local-typedefs
        -Wno-literal-suffix
        -Wno-narrowing
        -Wno-parentheses
        -Wno-unused-but-set-variable
        -Wno-unused-variable
        -Wno-char-subscripts
        -Wno-implicit-fallthrough
        -Wno-register
        -ffast-math
        -fPIC
        )
STRING(REPLACE ";" " " CXXFLAGS "${CXXFLAGS}")
STRING(APPEND CMAKE_CXX_FLAGS " ")
STRING(APPEND CMAKE_CXX_FLAGS ${CXXFLAGS})
MESSAGE(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
IF (NOT CMAKE_BUILD_TYPE)
    # Release With Debug Info
    SET(CMAKE_BUILD_TYPE RelWithDebInfo)
    MESSAGE(STATUS "CMAKE_CXX_FLAGS_RELWITHDEBINFO: " ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
ENDIF ()

# cuda
FIND_PACKAGE(CUDA REQUIRED)
MESSAGE(STATUS "CUDA_VERSION: ${CUDA_VERSION}")
MESSAGE(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
MESSAGE(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
MESSAGE(STATUS "CUDA_CUBLAS_LIBRARIES: ${CUDA_CUBLAS_LIBRARIES}")
INCLUDE_DIRECTORIES(/usr/local/cuda/include)
SET(CUDA_NVCC_FLAGS "-std=c++14" "-Xcompiler=-fno-gnu-unique")
SET(CMAKE_CUDA_STANDARD 14)

# tensorrt
IF (NOT DEFINED TENSORRT_DIR)
    SET(TENSORRT_DIR "/usr/local/TensorRT")
ENDIF ()
FIND_PATH(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include)
INCLUDE_DIRECTORIES(${TENSORRT_INCLUDE_DIR})
FIND_LIBRARY(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
FIND_LIBRARY(TENSORRT_LIBRARY_PARSERS nvparsers
        HINTS ${TENSORRT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
FIND_LIBRARY(TENSORRT_LIBRARY_INFER_PLUGINS nvinfer_plugin
        HINTS ${TENSORRT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
SET(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_PARSERS} ${TENSORRT_LIBRARY_INFER_PLUGINS})
MESSAGE(STATUS "TENSORRT_DIR: ${TENSORRT_DIR}")
MESSAGE(STATUS "TENSORRT_INCLUDE_DIR: ${TENSORRT_INCLUDE_DIR}")
MESSAGE(STATUS "TENSORRT_LIBRARY: ${TENSORRT_LIBRARY}")
FIND_PACKAGE_HANDLE_STANDARD_ARGS(
        TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY
)
IF (NOT TENSORRT_FOUND)
    MESSAGE(FATAL_ERROR "Failed to find TENSORRT library.")
ENDIF (NOT TENSORRT_FOUND)

# CUDA targets
IF (DEFINED GPU_ARCHS)
    MESSAGE(STATUS "GPU_ARCHS defined as ${GPU_ARCHS}. Generating CUDA code for SM ${GPU_ARCHS}")
    SEPARATE_ARGUMENTS(GPU_ARCHS)
ELSE()
    LIST(APPEND GPU_ARCHS
            61
            70
            75
            )
    STRING(REGEX MATCH "aarch64" IS_ARM "${TRT_PLATFORM_ID}")
    IF (IS_ARM)
        # Xavier (SM72) only supported for aarch64.
        LIST(APPEND GPU_ARCHS 72)
    ENDIF()

    IF (CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
        # Ampere GPU (SM80) support is only available in CUDA versions > 11.0
        LIST(APPEND GPU_ARCHS 80)
    ELSE()
        message(WARNING "Detected CUDA version is < 11.0. SM80 not supported.")
    ENDIF()

    MESSAGE(STATUS "GPU_ARCHS is not defined. Generating CUDA code for default SMs: ${GPU_ARCHS}")
ENDIF()
# Generate SASS for each architecture
FOREACH(arch ${GPU_ARCHS})
    SET(GENCODES "${GENCODES}" "-gencode arch=compute_${arch},code=sm_${arch}")
ENDFOREACH ()
SET(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCHS})
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "${GENCODES}")
MESSAGE(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
MESSAGE(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
MESSAGE(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

ADD_SUBDIRECTORY(src)