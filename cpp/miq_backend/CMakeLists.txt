cmake_minimum_required(VERSION 3.16)

project(hktex2quad_miq LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# MIQ was removed from newer libigl releases.  Pin the last stable libigl
# release that still ships the reference CoMISo implementation.
set(LIBIGL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LIBIGL_BUILD_TUTORIALS OFF CACHE BOOL "" FORCE)
set(LIBIGL_GLFW OFF CACHE BOOL "" FORCE)
set(LIBIGL_IMGUI OFF CACHE BOOL "" FORCE)
set(LIBIGL_OPENGL OFF CACHE BOOL "" FORCE)
set(LIBIGL_COPYLEFT_COMISO ON CACHE BOOL "" FORCE)
set(LIBIGL_COPYLEFT_CGAL OFF CACHE BOOL "" FORCE)
set(LIBIGL_COPYLEFT_TETGEN OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(
  libigl
  GIT_REPOSITORY https://github.com/libigl/libigl.git
  GIT_TAG fdaac01bcc52888994f7afd029dcc045dd408484
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(libigl)

add_executable(hktex2quad_miq main.cpp)
target_link_libraries(hktex2quad_miq PRIVATE igl::core igl_copyleft::comiso)

if(MSVC)
  target_compile_options(hktex2quad_miq PRIVATE /W4)
else()
  target_compile_options(hktex2quad_miq PRIVATE -Wall -Wextra -Wpedantic)
endif()
