#pragma once

#include <cstddef>
#include <span>
#include <type_traits>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace sparse_linalg::execution {

template<typename T>
struct SimdTraits {
    static constexpr bool is_vectorizable = false;
    static constexpr std::size_t vector_size = 1;
};

#if defined(__AVX2__)
template<>
struct SimdTraits<float> {
    static constexpr bool is_vectorizable = true;
    static constexpr std::size_t vector_size = 8;
    using vector_type = __m256;
    
    static vector_type load(const float* ptr) {
        return _mm256_loadu_ps(ptr);
    }
    
    static void store(float* ptr, vector_type val) {
        _mm256_storeu_ps(ptr, val);
    }
    
    static vector_type multiply(vector_type a, vector_type b) {
        return _mm256_mul_ps(a, b);
    }
    
    static vector_type add(vector_type a, vector_type b) {
        return _mm256_add_ps(a, b);
    }
    
    static vector_type set_zero() {
        return _mm256_setzero_ps();
    }
    
    static float reduce_sum(vector_type v) {
        __m128 high = _mm256_extractf128_ps(v, 1);
        __m128 low = _mm256_castps256_ps128(v);
        __m128 sum = _mm_add_ps(high, low);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    }
};

template<>
struct SimdTraits<double> {
    static constexpr bool is_vectorizable = true;
    static constexpr std::size_t vector_size = 4;
    using vector_type = __m256d;
    
    static vector_type load(const double* ptr) {
        return _mm256_loadu_pd(ptr);
    }
    
    static void store(double* ptr, vector_type val) {
        _mm256_storeu_pd(ptr, val);
    }
    
    static vector_type multiply(vector_type a, vector_type b) {
        return _mm256_mul_pd(a, b);
    }
    
    static vector_type add(vector_type a, vector_type b) {
        return _mm256_add_pd(a, b);
    }
    
    static vector_type set_zero() {
        return _mm256_setzero_pd();
    }
    
    static double reduce_sum(vector_type v) {
        __m128d high = _mm256_extractf128_pd(v, 1);
        __m128d low = _mm256_castpd256_pd128(v);
        __m128d sum = _mm_add_pd(high, low);
        sum = _mm_hadd_pd(sum, sum);
        return _mm_cvtsd_f64(sum);
    }
};
#endif

} // namespace sparse_linalg::execution