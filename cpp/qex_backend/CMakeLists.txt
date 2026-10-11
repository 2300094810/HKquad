cmake_minimum_required(VERSION 3.16)
project(hktex2quad_qex LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FetchContent)
FetchContent_Declare(openmesh
  GIT_REPOSITORY https://github.com/autodesk-forks/OpenMesh.git
  GIT_TAG 8e949df8fb00673a2af9ca2b0a513d52e700ec94)
FetchContent_Declare(qex
  GIT_REPOSITORY https://github.com/hcebke/libQEx.git
  GIT_TAG 517dcaa0cc87646baa89e52cfc8e23766776f6d5)
FetchContent_GetProperties(openmesh)
if(NOT openmesh_POPULATED)
  FetchContent_Populate(openmesh)
endif()
FetchContent_GetProperties(qex)
if(NOT qex_POPULATED)
  FetchContent_Populate(qex)
endif()
# Build the core only, without upstream's legacy GUI/install CMake machinery.
file(GLOB_RECURSE OM_CORE CONFIGURE_DEPENDS ${openmesh_SOURCE_DIR}/src/OpenMesh/Core/*.cc)
list(FILTER OM_CORE EXCLUDE REGEX "T\\.cc$")
list(FILTER OM_CORE EXCLUDE REGEX "/Templates/")
add_library(openmesh_core STATIC ${OM_CORE})
target_include_directories(openmesh_core PUBLIC ${openmesh_SOURCE_DIR}/src)
add_executable(hktex2quad_qex main.cpp
  ${qex_SOURCE_DIR}/interfaces/c/qex.cc
  ${qex_SOURCE_DIR}/src/predicates.c
  ${qex_SOURCE_DIR}/src/MeshExtractor.cc)
target_include_directories(hktex2quad_qex PRIVATE ${qex_SOURCE_DIR}/interfaces/c)
target_link_libraries(hktex2quad_qex PRIVATE openmesh_core)
target_compile_definitions(hktex2quad_qex PRIVATE QEX_EXPORT_SYMBOLS)
# Exact predicates require standard precision, not x87 extended intermediates.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i.86")
  target_compile_options(hktex2quad_qex PRIVATE -msse -mfpmath=sse -fno-fast-math)
endif()
