cmake_minimum_required(VERSION 3.14)
project(llama_flutter_android_plugin LANGUAGES C CXX)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(GGML_RPC ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)

find_package(Vulkan QUIET)
if (Vulkan_FOUND AND Vulkan_GLSLC_EXECUTABLE)
    message(STATUS "Vulkan SDK found, enabling GGML_VULKAN for Windows")
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
else()
    message(STATUS "Vulkan SDK not found, building Windows llama with CPU + RPC")
    set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
endif()

set(LLAMA_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../android/src/main/cpp/llama.cpp")
add_subdirectory("${LLAMA_SRC_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)

set(PLUGIN_NAME llama_flutter_android_plugin)

add_library(${PLUGIN_NAME} SHARED
  "include/llama_flutter_android/llama_flutter_android_plugin_c_api.h"
  "llama_flutter_android_plugin_c_api.cpp"
  "llama_flutter_android_plugin.h"
  "llama_flutter_android_plugin.cpp"
  "../src/llama_desktop_core.h"
  "../src/llama_desktop_core.cpp"
)

apply_standard_settings(${PLUGIN_NAME})

target_compile_options(${PLUGIN_NAME} PRIVATE /EHsc)
target_compile_definitions(${PLUGIN_NAME} PRIVATE
  FLUTTER_PLUGIN_IMPL
  _CRT_SECURE_NO_WARNINGS
  WIN32_LEAN_AND_MEAN
  NOMINMAX
  _HAS_EXCEPTIONS=1
)

target_include_directories(${PLUGIN_NAME} INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_include_directories(${PLUGIN_NAME} PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src"
  "${LLAMA_SRC_DIR}/include"
  "${LLAMA_SRC_DIR}/ggml/include"
  "${LLAMA_SRC_DIR}/tools/mtmd"
)

target_link_libraries(${PLUGIN_NAME} PRIVATE
  flutter
  flutter_wrapper_plugin
  llama
  ggml
  ggml-rpc
  mtmd
  ws2_32
  psapi
)

set(llama_flutter_android_bundled_libraries
  $<TARGET_FILE:${PLUGIN_NAME}>
  PARENT_SCOPE
)
