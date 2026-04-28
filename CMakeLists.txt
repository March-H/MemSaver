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
find_package(Torch QUIET)

if(NOT Torch_FOUND)
  execute_process(
    COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE MEMSAVER_TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(MEMSAVER_TORCH_CMAKE_PREFIX)
    list(PREPEND CMAKE_PREFIX_PATH "${MEMSAVER_TORCH_CMAKE_PREFIX}")
    find_package(Torch QUIET)
  endif()
endif()

if(Torch_FOUND)
  separate_arguments(MEMSAVER_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  execute_process(
    COMMAND python -c "import importlib, os; dirs=[]; modules=('nvidia.cuda_runtime','nvidia.cuda_cupti','nvidia.cuda_nvrtc'); [dirs.append(os.path.join(os.path.dirname(importlib.import_module(module).__file__), 'lib')) for module in modules if importlib.util.find_spec(module)]; print(';'.join(dirs))"
    OUTPUT_VARIABLE MEMSAVER_TORCH_CUDA_LIBRARY_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(MEMSAVER_TORCH_CUDA_LIBRARY_DIRS)
    find_library(
      MEMSAVER_TORCH_CUDART_LIBRARY
      NAMES libcudart.so.12 cudart
      HINTS ${MEMSAVER_TORCH_CUDA_LIBRARY_DIRS}
      NO_DEFAULT_PATH
    )
    find_library(
      MEMSAVER_TORCH_CUPTI_LIBRARY
      NAMES libcupti.so.12 cupti
      HINTS ${MEMSAVER_TORCH_CUDA_LIBRARY_DIRS}
      NO_DEFAULT_PATH
    )
    find_library(
      MEMSAVER_TORCH_NVRTC_LIBRARY
      NAMES libnvrtc.so.12 nvrtc
      HINTS ${MEMSAVER_TORCH_CUDA_LIBRARY_DIRS}
      NO_DEFAULT_PATH
    )
  endif()
endif()

set(MEMSAVER_CUDART_LINK_LIBRARY CUDA::cudart)
if(MEMSAVER_TORCH_CUDART_LIBRARY)
  set(MEMSAVER_CUDART_LINK_LIBRARY ${MEMSAVER_TORCH_CUDART_LIBRARY})
endif()

set(MEMSAVER_CUPTI_LINK_LIBRARY CUDA::cupti)
if(MEMSAVER_TORCH_CUPTI_LIBRARY)
  set(MEMSAVER_CUPTI_LINK_LIBRARY ${MEMSAVER_TORCH_CUPTI_LIBRARY})
endif()

set(MEMSAVER_TORCH_EXTRA_LINK_LIBRARIES)
if(MEMSAVER_TORCH_NVRTC_LIBRARY)
  list(APPEND MEMSAVER_TORCH_EXTRA_LINK_LIBRARIES ${MEMSAVER_TORCH_NVRTC_LIBRARY})
endif()
if(MEMSAVER_TORCH_CUDART_LIBRARY)
  list(APPEND MEMSAVER_TORCH_EXTRA_LINK_LIBRARIES ${MEMSAVER_TORCH_CUDART_LIBRARY})
endif()
if(MEMSAVER_TORCH_CUPTI_LIBRARY)
  list(APPEND MEMSAVER_TORCH_EXTRA_LINK_LIBRARIES ${MEMSAVER_TORCH_CUPTI_LIBRARY})
endif()

set(MEMSAVER_SOURCES
  src/internal/context_impl.cpp
  src/entrypoint.cpp
)

if(MEMSAVER_FATAL_ON_ERROR)
  set(MEMSAVER_FATAL_ON_ERROR_VALUE 1)
else()
  set(MEMSAVER_FATAL_ON_ERROR_VALUE 0)
endif()

add_library(memsaver ${MEMSAVER_SOURCES})
target_compile_features(memsaver PUBLIC cxx_std_17)
target_compile_definitions(memsaver
  PRIVATE MEMSAVER_FATAL_ON_ERROR=${MEMSAVER_FATAL_ON_ERROR_VALUE})
target_include_directories(memsaver
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_options(memsaver PRIVATE ${MEMSAVER_TORCH_CXX_FLAGS})
target_link_libraries(memsaver PUBLIC CUDA::cuda_driver ${MEMSAVER_CUDART_LINK_LIBRARY})
target_link_libraries(memsaver PRIVATE dl ${TORCH_LIBRARIES} ${MEMSAVER_TORCH_EXTRA_LINK_LIBRARIES})
set_target_properties(memsaver PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(Torch_FOUND)
  set_target_properties(memsaver PROPERTIES
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib;${MEMSAVER_TORCH_CUDA_LIBRARY_DIRS}")
endif()

install(TARGETS
    memsaver
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

  if(Torch_FOUND)
    add_executable(memsaver_torch_basic_test tests/basic_test.cpp)
    target_compile_features(memsaver_torch_basic_test PUBLIC cxx_std_17)
    target_compile_definitions(memsaver_torch_basic_test
      PRIVATE MEMSAVER_FATAL_ON_ERROR=${MEMSAVER_FATAL_ON_ERROR_VALUE})
    target_compile_options(memsaver_torch_basic_test PRIVATE ${MEMSAVER_TORCH_CXX_FLAGS})
    target_include_directories(memsaver_torch_basic_test
      PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(memsaver_torch_basic_test
      PRIVATE
        memsaver
        ${TORCH_LIBRARIES}
        ${MEMSAVER_TORCH_EXTRA_LINK_LIBRARIES}
        CUDA::cuda_driver
        ${MEMSAVER_CUDART_LINK_LIBRARY}
        ${MEMSAVER_CUPTI_LINK_LIBRARY}
    )
    if(DEFINED TORCH_INSTALL_PREFIX)
      set_target_properties(memsaver_torch_basic_test PROPERTIES
        BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib;${MEMSAVER_TORCH_CUDA_LIBRARY_DIRS}")
    endif()
    add_test(NAME memsaver_torch_basic_test COMMAND memsaver_torch_basic_test)

    add_executable(memsaver_torch_arena_test tests/arena_test.cpp)
    target_compile_features(memsaver_torch_arena_test PUBLIC cxx_std_17)
    target_compile_definitions(memsaver_torch_arena_test
      PRIVATE MEMSAVER_FATAL_ON_ERROR=${MEMSAVER_FATAL_ON_ERROR_VALUE})
    target_compile_options(memsaver_torch_arena_test PRIVATE ${MEMSAVER_TORCH_CXX_FLAGS})
    target_include_directories(memsaver_torch_arena_test
      PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(memsaver_torch_arena_test
      PRIVATE
        memsaver
        ${TORCH_LIBRARIES}
        ${MEMSAVER_TORCH_EXTRA_LINK_LIBRARIES}
        CUDA::cuda_driver
        ${MEMSAVER_CUDART_LINK_LIBRARY}
        ${MEMSAVER_CUPTI_LINK_LIBRARY}
    )
    if(DEFINED TORCH_INSTALL_PREFIX)
      set_target_properties(memsaver_torch_arena_test PROPERTIES
        BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib;${MEMSAVER_TORCH_CUDA_LIBRARY_DIRS}")
    endif()
    add_test(NAME memsaver_torch_arena_test COMMAND memsaver_torch_arena_test)
  endif()
endif()
