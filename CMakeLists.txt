cmake_minimum_required(VERSION 3.10)
project(Pilot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4 /EHsc)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# --- 查找库 ---
find_package(FFmpeg REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Threads REQUIRED)

# --- ONNX Runtime 手动配置 ---
set(ORT_DIR "D:/onnxruntime-win-x64-gpu-1.24.1")

find_library(ORT_LIB onnxruntime 
    PATHS "${ORT_DIR}/lib" 
    NO_DEFAULT_PATH)

find_library(ORT_CUDA_PROVIDER_LIB onnxruntime_providers_cuda 
    PATHS "${ORT_DIR}/lib" 
    NO_DEFAULT_PATH)

# --- 定义程序资源 ---
set(COMMON_SOURCES
    "service/service.cpp"
    "utils/WebServerUtils.cpp"
    "feature/feature.cpp"
)

set(APP_SOURCES "service/entry.cpp" ${COMMON_SOURCES})
set(TEST_SOURCES "test/test_rtspPlay.cpp" ${COMMON_SOURCES})

# 收集所有头文件
file(GLOB_RECURSE HEADER_FILES "service/*.hpp" "service/*.h" "utils/*.h" "feature/*.hpp")

# --- 汇总依赖 ---
set(ALL_LIBS 
    ${OpenCV_LIBS} 
    ${TORCH_LIBRARIES} 
    ${FFMPEG_LIBRARIES}
    ${ORT_LIB} 
    ${ORT_CUDA_PROVIDER_LIB}
    Python3::Python 
    Threads::Threads
)

set(ALL_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    "${ORT_DIR}/include"  # 加入 ORT 头文件路径
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIRS}
)

# --- 创建目标 ---
# 主程序
add_executable(PilotApp ${APP_SOURCES} ${HEADER_FILES})
set_target_properties(PilotApp PROPERTIES OUTPUT_NAME "pilot")
target_include_directories(PilotApp PUBLIC ${ALL_INCLUDES})
target_link_libraries(PilotApp PRIVATE ${ALL_LIBS})

# 测试程序
add_executable(Tst ${TEST_SOURCES} ${HEADER_FILES})
set_target_properties(Tst PROPERTIES OUTPUT_NAME "test_runner")
target_include_directories(Tst PUBLIC ${ALL_INCLUDES})
target_link_libraries(Tst PRIVATE ${ALL_LIBS})

# 自动拷贝DLL
set(ORT_DLLS 
    "${ORT_DIR}/lib/onnxruntime.dll"
    "${ORT_DIR}/lib/onnxruntime_providers_cuda.dll"
)

foreach(target PilotApp Tst)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ORT_DLLS}
        $<TARGET_FILE_DIR:${target}>
        COMMENT "Copying ONNX Runtime DLLs to output directory..."
    )
endforeach()



# ---Torch的特殊处理---
# 如果是在 Linux 下使用 LibTorch，需要下面这行来解决链接问题
#if (NOT MSVC)
#   set_property(TARGET PilotApp PROPERTY CXX_STANDARD 14) # Torch 常用 14
#   target_compile_options(PilotApp PUBLIC ${TORCH_CXX_FLAGS})
#   target_compile_options(Tst PUBLIC ${TORCH_CXX_FLAGS})
#endif()