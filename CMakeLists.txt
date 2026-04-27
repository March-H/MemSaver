cmake_minimum_required(VERSION 3.20)

project(MemSaver VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(MEMSAVER_FATAL_ON_ERROR "Abort process on internal invariant violations." ON)
option(MEMSAVER_BUILD_TESTS "Build C++ tests." ON)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "MemSaver currently supports Linux only.")
endif()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" MEMSAVER_SYSTEM_PROCESSOR)
if(NOT MEMSAVER_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
  message(FATAL_ERROR "MemSaver currently supports Linux x86_64 only.")
endif()

find_package(CUDAToolkit REQUIRED)

set(MEMSAVER_CORE_SOURCES
  src/internal/context_impl.cpp
  src/entrypoint.cpp
)

if(MEMSAVER_FATAL_ON_ERROR)
  set(MEMSAVER_FATAL_ON_ERROR_VALUE 1)
else()
  set(MEMSAVER_FATAL_ON_ERROR_VALUE 0)
endif()

add_library(memsaver_core_shared SHARED ${MEMSAVER_CORE_SOURCES})
add_library(memsaver_core_static STATIC ${MEMSAVER_CORE_SOURCES})

foreach(target_name IN ITEMS memsaver_core_shared memsaver_core_static)
  target_compile_features(${target_name} PUBLIC cxx_std_17)
  target_compile_definitions(${target_name}
    PRIVATE MEMSAVER_FATAL_ON_ERROR=${MEMSAVER_FATAL_ON_ERROR_VALUE})
  target_include_directories(${target_name}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CUDAToolkit_INCLUDE_DIRS}
  )
  target_link_libraries(${target_name} PUBLIC CUDA::cuda_driver CUDA::cudart)
  target_link_libraries(${target_name} PRIVATE dl)
  set_target_properties(${target_name} PROPERTIES
    OUTPUT_NAME memsaver_core
    POSITION_INDEPENDENT_CODE ON
  )
endforeach()

install(TARGETS
    memsaver_core_shared
    memsaver_core_static
  EXPORT MemSaverTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT MemSaverTargets
  FILE MemSaverTargets.cmake
  NAMESPACE MemSaver::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MemSaver
)

configure_package_config_file(
  cmake/MemSaverConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/MemSaverConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MemSaver
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/MemSaverConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/MemSaverConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/MemSaverConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MemSaver
)

if(MEMSAVER_BUILD_TESTS)
  include(CTest)
endif()
