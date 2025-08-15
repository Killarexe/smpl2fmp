cmake_minimum_required(VERSION 3.10)
project(smpl2fmp VERSION 1.0.0 LANGUAGES CXX DESCRIPTION "A software that converts samples into FM patches")

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED fftw3)

# Adding our source files
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp") # Define PROJECT_SOURCES as a list of all source files
set(PROJECT_INCLUDE "${CMAKE_CURRENT_LIST_DIR}/src/") # Define PROJECT_INCLUDE to be the path to the include directory of the project

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# AudioFile compile settings
set(BUILD_TESTS OFF)
set(BUILD_EXAMPLES OFF)

include(FetchContent)

FetchContent_Declare(
    cli11
    QUIET
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG "main"
)

FetchContent_Declare(
  AudioFile
  QUIET
  GIT_REPOSITORY https://github.com/adamstark/AudioFile.git
  GIT_TAG 1.1.3
)

FetchContent_Declare(
  pgbar_proj
  QUIET
  GIT_REPOSITORY https://github.com/Konvt/pgbar.git
  GIT_TAG "main"
)

FetchContent_MakeAvailable(cli11)
FetchContent_MakeAvailable(AudioFile)
FetchContent_MakeAvailable(pgbar_proj)

# Declaring our executable
add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_INCLUDE})
target_include_directories(${PROJECT_NAME} PRIVATE ${FFTW_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${cli11_})
target_link_libraries(${PROJECT_NAME} PRIVATE CLI11::CLI11)
target_link_libraries(${PROJECT_NAME} PRIVATE AudioFile)
target_link_libraries(${PROJECT_NAME} PRIVATE pgbar)
target_link_libraries(${PROJECT_NAME} PRIVATE ${FFTW_LIBRARIES})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()

# Compile warnings
target_compile_options(${PROJECT_NAME} PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -O3 -march=native -ffast-math -fopenmp -DNDEBUG -fno-stack-protector -march=native -Wpedantic>
)
