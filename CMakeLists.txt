cmake_minimum_required(VERSION 3.20)

project(MemSaver VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(MEMSAVER_FATAL_ON_ERROR "Abort process on internal invariant violations." ON)
option(MEMSAVER_BUILD_TESTS "Build C++ tests." ON)
option(MEMSAVER_BUILD_TORCH_TESTS "Build libtorch-based C++ tests." ON)
set(MEMSAVER_TORCH_CMAKE_PREFIX ""
    CACHE PATH "Optional path to torch/share/cmake for find_package(Torch)")

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
  src/internal/original_cuda_api.cpp
  src/memsaver_c.cpp
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

add_library(memsaver_preload SHARED
  src/internal/context_impl.cpp
  src/internal/original_cuda_api.cpp
  src/preload.cpp
)
target_compile_features(memsaver_preload PUBLIC cxx_std_17)
target_compile_definitions(memsaver_preload
  PRIVATE MEMSAVER_FATAL_ON_ERROR=${MEMSAVER_FATAL_ON_ERROR_VALUE})
target_include_directories(memsaver_preload
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(memsaver_preload PUBLIC CUDA::cuda_driver CUDA::cudart)
target_link_libraries(memsaver_preload PRIVATE dl)
set_target_properties(memsaver_preload PROPERTIES
  OUTPUT_NAME memsaver_preload
  POSITION_INDEPENDENT_CODE ON
)

install(TARGETS
    memsaver_core_shared
    memsaver_core_static
    memsaver_preload
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

  add_executable(memsaver_core_tests tests/cpp/core_tests.cpp)
  target_compile_features(memsaver_core_tests PRIVATE cxx_std_17)
  target_include_directories(memsaver_core_tests PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(memsaver_core_tests PRIVATE memsaver_core_shared CUDA::cudart)
  add_test(NAME memsaver_core_tests COMMAND memsaver_core_tests)

  add_executable(memsaver_preload_smoke tests/cpp/preload_smoke.cpp)
  target_compile_features(memsaver_preload_smoke PRIVATE cxx_std_17)
  target_include_directories(memsaver_preload_smoke PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(memsaver_preload_smoke PRIVATE CUDA::cuda_driver CUDA::cudart dl)
  add_test(
    NAME memsaver_preload_smoke
    COMMAND ${CMAKE_COMMAND} -E env
      LD_PRELOAD=$<TARGET_FILE:memsaver_preload>
      MEMSAVER_INIT_ENABLE=0
      MEMSAVER_INIT_ENABLE_CPU_BACKUP=0
      MEMSAVER_INIT_ALLOCATION_MODE=normal
      $<TARGET_FILE:memsaver_preload_smoke>
  )

  if(MEMSAVER_BUILD_TORCH_TESTS)
    set(_memsaver_torch_prefixes "")
    if(MEMSAVER_TORCH_CMAKE_PREFIX)
      list(APPEND _memsaver_torch_prefixes "${MEMSAVER_TORCH_CMAKE_PREFIX}")
    endif()

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
      execute_process(
        COMMAND ${Python3_EXECUTABLE} -c
          "import torch; print(torch.utils.cmake_prefix_path)"
        RESULT_VARIABLE MEMSAVER_TORCH_PREFIX_RESULT
        OUTPUT_VARIABLE MEMSAVER_TORCH_PREFIX_FROM_PYTHON
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      if(MEMSAVER_TORCH_PREFIX_RESULT EQUAL 0 AND MEMSAVER_TORCH_PREFIX_FROM_PYTHON)
        list(APPEND _memsaver_torch_prefixes "${MEMSAVER_TORCH_PREFIX_FROM_PYTHON}")
      endif()
    endif()

    list(REMOVE_DUPLICATES _memsaver_torch_prefixes)
    foreach(_prefix IN LISTS _memsaver_torch_prefixes)
      list(PREPEND CMAKE_PREFIX_PATH "${_prefix}")
    endforeach()

    find_package(Torch QUIET)
    if(Torch_FOUND)
      set(MEMSAVER_TORCH_LIBS ${TORCH_LIBRARIES})
      set(MEMSAVER_PYTHON_CUDART_LIB "")
      set(MEMSAVER_PYTHON_CUPTI_LIB "")
      if(Python3_Interpreter_FOUND)
        execute_process(
          COMMAND ${Python3_EXECUTABLE} -c
            "import pathlib, nvidia.cuda_runtime.lib as m; p = pathlib.Path(m.__file__).with_name('libcudart.so.12'); print(p if p.exists() else '')"
          RESULT_VARIABLE MEMSAVER_CUDART_RESULT
          OUTPUT_VARIABLE MEMSAVER_PYTHON_CUDART_LIB
          OUTPUT_STRIP_TRAILING_WHITESPACE
          ERROR_QUIET
        )
        if(NOT MEMSAVER_CUDART_RESULT EQUAL 0)
          set(MEMSAVER_PYTHON_CUDART_LIB "")
        endif()

        execute_process(
          COMMAND ${Python3_EXECUTABLE} -c
            "import pathlib, nvidia.cuda_cupti.lib as m; p = pathlib.Path(m.__file__).with_name('libcupti.so.12'); print(p if p.exists() else '')"
          RESULT_VARIABLE MEMSAVER_CUPTI_RESULT
          OUTPUT_VARIABLE MEMSAVER_PYTHON_CUPTI_LIB
          OUTPUT_STRIP_TRAILING_WHITESPACE
          ERROR_QUIET
        )
        if(NOT MEMSAVER_CUPTI_RESULT EQUAL 0)
          set(MEMSAVER_PYTHON_CUPTI_LIB "")
        endif()
      endif()

      if(MEMSAVER_PYTHON_CUDART_LIB)
        list(FILTER MEMSAVER_TORCH_LIBS EXCLUDE REGEX ".*libcudart\\.so(\\..*)?$")
        list(APPEND MEMSAVER_TORCH_LIBS "${MEMSAVER_PYTHON_CUDART_LIB}")
      endif()
      if(MEMSAVER_PYTHON_CUPTI_LIB)
        list(APPEND MEMSAVER_TORCH_LIBS "${MEMSAVER_PYTHON_CUPTI_LIB}")
      endif()
      list(REMOVE_DUPLICATES MEMSAVER_TORCH_LIBS)

      add_executable(memsaver_torch_basic_test tests/cpp/basic_test.cpp)
      target_compile_features(memsaver_torch_basic_test PRIVATE cxx_std_17)
      target_include_directories(memsaver_torch_basic_test PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
      )
      target_link_libraries(memsaver_torch_basic_test PRIVATE
        ${MEMSAVER_TORCH_LIBS}
        CUDA::cuda_driver
        CUDA::cupti
        dl
      )
      separate_arguments(MEMSAVER_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
      target_compile_options(memsaver_torch_basic_test PRIVATE ${MEMSAVER_TORCH_CXX_FLAGS})

      add_test(
        NAME memsaver_torch_basic_test
        COMMAND ${CMAKE_COMMAND} -E env
          LD_PRELOAD=$<TARGET_FILE:memsaver_preload>
          MEMSAVER_INIT_ENABLE=0
          MEMSAVER_INIT_ENABLE_CPU_BACKUP=0
          MEMSAVER_INIT_ALLOCATION_MODE=normal
          $<TARGET_FILE:memsaver_torch_basic_test>
      )
    else()
      message(WARNING
        "Torch C++ package not found; skipping memsaver_torch_basic_test. "
        "Set MEMSAVER_TORCH_CMAKE_PREFIX to torch/share/cmake if needed.")
    endif()
  endif()
endif()
