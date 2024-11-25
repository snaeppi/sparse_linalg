add_library(doctest INTERFACE)
target_include_directories(doctest INTERFACE "${CMAKE_SOURCE_DIR}/external")
add_library(doctest::doctest ALIAS doctest)

add_executable(sparse_linalg_tests
    main.cpp
    src/sparse_matrix_test.cpp
    src/matrix_ops_test.cpp
    src/thread_pool_test.cpp
)

target_link_libraries(sparse_linalg_tests
    PRIVATE
        sparse_linalg
        doctest::doctest
)

add_test(NAME sparse_linalg_tests COMMAND sparse_linalg_tests)

set_tests_properties(sparse_linalg_tests 
    PROPERTIES
        LABELS "unit"
)

set_tests_properties(sparse_linalg_tests
    PROPERTIES
        TIMEOUT 30  # seconds
)