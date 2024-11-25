# sparse_linalg

A learning project implementing sparse linear algebra operations in modern C++20.

## Features

- Header-only implementation exploring various C++20 features
- Basic sparse matrix storage using the Compressed Sparse Row (CSR) format
- Custom thread pool implementation
- SIMD operations using AVX2 intrinsics
- Test suite using doctest
- Performance benchmarking using Google Benchmark

## Requirements

- C++20 compliant compiler (tested with GCC 11+)
- CMake 3.20 or newer
- AVX2 support for SIMD implementations (optional)

## Building

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## Project Status

Currently, the library implements sparse matrix-vector multiplication with different optimization strategies.

Near-term development priorities:
- Implementation of basic sparse matrix operations:
  - Matrix-matrix multiplication
  - Addition and subtraction
  - Transpose
  - Element-wise operations
- Support for different numeric types

Longer-term goals:
- Implementing iterative solvers (Conjugate Gradient, GMRES)
- Exploring different sparse matrix formats (COO, CSC, Block CSR)
- Development of a task-based parallelism system
- Adding matrix reordering algorithms to improve cache efficiency
- Implementing matrix decomposition methods (LU, Cholesky)

The project serves primarily as a platform for learning about numerical algorithms, parallel programming patterns, and modern C++ features.