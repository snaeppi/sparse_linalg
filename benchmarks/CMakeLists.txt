include(FetchContent)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.0
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(benchmark)

add_executable(sparse_linalg_benchmarks
    src/matrix_ops_bench.cpp
)

target_link_libraries(sparse_linalg_benchmarks
    PRIVATE
        sparse_linalg
        benchmark::benchmark
        benchmark::benchmark_main
)