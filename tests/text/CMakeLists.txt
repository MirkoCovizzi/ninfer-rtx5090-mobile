cmake_minimum_required(VERSION 3.28)
project(ninfer_text_tests LANGUAGES C CXX)

# Build the actual native renderer for CPU-only CI, without configuring CUDA or media libraries.
get_filename_component(NINFER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
find_package(Threads REQUIRED)
add_subdirectory("${NINFER_ROOT}/third_party/llama-jinja" llama-jinja)
target_sources(ninfer_jinja PRIVATE
  "${NINFER_ROOT}/src/text/jinja.cpp"
  "${NINFER_ROOT}/src/text/unicode.cpp"
  "${NINFER_ROOT}/third_party/utf8proc/utf8proc.c")
target_include_directories(ninfer_jinja PUBLIC
  "${NINFER_ROOT}/src"
  "${NINFER_ROOT}/third_party")
target_compile_features(ninfer_jinja PUBLIC cxx_std_20)

add_executable(ninfer_jinja_test test_jinja.cpp)
target_link_libraries(ninfer_jinja_test PRIVATE ninfer_jinja Threads::Threads)
enable_testing()
add_test(NAME ninfer_jinja_test COMMAND ninfer_jinja_test)
