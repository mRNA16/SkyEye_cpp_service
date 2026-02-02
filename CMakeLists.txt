cmake_minimum_required(VERSION 3.10)
project(Pilot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4 /EHsc)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# --- 查找库 ---
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Threads REQUIRED)

# --- 定义程序 ---
set(COMMON_SOURCES
    "service/service.cpp"
    "utils/WebServerUtils.cpp"
)

set(APP_SOURCES "service/entry.cpp" ${COMMON_SOURCES})
set(TEST_SOURCES "test/test_rtspPlay.cpp" ${COMMON_SOURCES})

# 收集所有头文件
file(GLOB_RECURSE HEADER_FILES "service/*.hpp" "service/*.h" "utils/*.h")

# 创建目标
# 主程序
add_executable(PilotApp ${APP_SOURCES} ${HEADER_FILES})
set_target_properties(PilotApp PROPERTIES OUTPUT_NAME "pilot")

# 测试程序
add_executable(Tst ${TEST_SOURCES} ${HEADER_FILES})
set_target_properties(Tst PROPERTIES OUTPUT_NAME "test_runner")

set(ALL_LIBS 
    ${OpenCV_LIBS} 
    ${TORCH_LIBRARIES} 
    Python3::Python 
    Threads::Threads
)

set(ALL_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

# 应用到 PilotApp
target_include_directories(PilotApp PUBLIC ${ALL_INCLUDES})
target_link_libraries(PilotApp PRIVATE ${ALL_LIBS})

# 应用到 Tst
target_include_directories(Tst PUBLIC ${ALL_INCLUDES})
target_link_libraries(Tst PRIVATE ${ALL_LIBS})



# ---Torch的特殊处理---
# 如果是在 Linux 下使用 LibTorch，需要下面这行来解决链接问题
#if (NOT MSVC)
#   set_property(TARGET PilotApp PROPERTY CXX_STANDARD 14) # Torch 常用 14
#   target_compile_options(PilotApp PUBLIC ${TORCH_CXX_FLAGS})
#   target_compile_options(Tst PUBLIC ${TORCH_CXX_FLAGS})
#endif()