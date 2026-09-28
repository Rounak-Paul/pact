cmake_minimum_required(VERSION 3.21)

project(pact VERSION 0.1.0 LANGUAGES C)

include(CTest)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
find_package(Threads REQUIRED)

option(PACT_BUILD_BENCHMARKS "Build PACT benchmarks" OFF)

find_path(PACT_VULKAN_INCLUDE_DIR NAMES vulkan/vulkan.h REQUIRED)

add_library(pact
    pact/src/core/runtime.c
    pact/src/core/thread.c
    pact/src/core/cpu_pool.c
    pact/src/core/tensor.c
    pact/src/core/graph.c
    pact/src/core/planner.c
    pact/src/backend/cpu/cpu.c
    pact/src/backend/vulkan/loader.c
    pact/src/backend/vulkan/vulkan.c
    pact/src/backend/vulkan/vulkan_compute.c
)
add_library(pact::pact ALIAS pact)

target_compile_features(pact PUBLIC c_std_11)
set_target_properties(pact PROPERTIES
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS NO
    C_VISIBILITY_PRESET hidden
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME pact
)

target_include_directories(pact
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        pact/src/core
        ${PACT_VULKAN_INCLUDE_DIR}
)

target_link_libraries(pact PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)

if(BUILD_SHARED_LIBS)
    target_compile_definitions(pact PUBLIC PACT_SHARED=1)
    target_compile_definitions(pact PRIVATE PACT_BUILDING_LIBRARY=1)
endif()

if(MSVC)
    target_compile_options(pact PRIVATE /W4)
else()
    target_compile_options(pact PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(BUILD_TESTING)
    add_executable(pact_runtime_test tests/runtime.c)
    target_link_libraries(pact_runtime_test PRIVATE pact::pact)
    add_test(NAME pact.runtime COMMAND pact_runtime_test)

    add_executable(pact_tensor_test tests/tensor.c)
    target_link_libraries(pact_tensor_test PRIVATE pact::pact)
    add_test(NAME pact.tensor COMMAND pact_tensor_test)

    if(NOT BUILD_SHARED_LIBS)
        add_executable(pact_vulkan_backend_test tests/vulkan_backend.c)
        target_include_directories(pact_vulkan_backend_test PRIVATE pact/src/core)
        target_link_libraries(pact_vulkan_backend_test PRIVATE pact::pact)
        add_test(NAME pact.vulkan_backend COMMAND pact_vulkan_backend_test)

        add_executable(pact_planner_test tests/planner.c)
        target_include_directories(pact_planner_test PRIVATE pact/src/core)
        target_link_libraries(pact_planner_test PRIVATE pact::pact)
        add_test(NAME pact.planner COMMAND pact_planner_test)
    endif()

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_library(pact_test_block_vulkan MODULE tests/block_vulkan.c)
        target_link_libraries(pact_test_block_vulkan PRIVATE ${CMAKE_DL_LIBS})
        add_test(NAME pact.tensor.no_vulkan
            COMMAND ${CMAKE_COMMAND} -E env
                "LD_PRELOAD=$<TARGET_FILE:pact_test_block_vulkan>"
                $<TARGET_FILE:pact_tensor_test>
        )
        add_test(NAME pact.runtime.no_vulkan
            COMMAND ${CMAKE_COMMAND} -E env
                "LD_PRELOAD=$<TARGET_FILE:pact_test_block_vulkan>"
                "PACT_TEST_EXPECT_NO_VULKAN=1"
                $<TARGET_FILE:pact_runtime_test>
        )
    endif()
endif()

install(TARGETS pact EXPORT pactTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT pactTargets NAMESPACE pact:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pact)

configure_package_config_file(
    cmake/pactConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/pactConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pact
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/pactConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/pactConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/pactConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pact
)

if(PACT_BUILD_BENCHMARKS)
    add_executable(pact_add_benchmark benchmarks/add.c)
    target_link_libraries(pact_add_benchmark PRIVATE pact::pact)
endif()
