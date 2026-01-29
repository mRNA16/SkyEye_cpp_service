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
set(SOURCE_FILES 
    "service/entry.cpp"
    "service/service.cpp"
 "utils/WebServerUtils.cpp")

# 收集所有头文件
file(GLOB_RECURSE HEADER_FILES "service/*.hpp" "service/*.h" "utils/*.h")

# 创建目标
add_executable(PilotApp ${SOURCE_FILES} ${HEADER_FILES})

# --- 配置包含路径 (IntelliSense 的核心) ---
target_include_directories(PilotApp PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_LIBRARIES}
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

# --- 链接库 ---
target_link_libraries(PilotApp PRIVATE 
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    Python3::Python
    Threads::Threads
)

set_target_properties(PilotApp PROPERTIES OUTPUT_NAME "pilot")
