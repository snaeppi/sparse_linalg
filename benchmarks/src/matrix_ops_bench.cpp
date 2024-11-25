#include <benchmark/benchmark.h>
#include <sparse_linalg/core/sparse_matrix.hpp>
#include <sparse_linalg/core/matrix_ops.hpp>
#include <sparse_linalg/execution/thread_pool.hpp>
#include <random>
#include <memory>

using namespace sparse_linalg;

namespace {

template<typename T>
class BenchmarkFixture : public benchmark::Fixture {
protected:
    void SetUp(benchmark::State& state) override {
        SetUpImpl(state);
    }
    
    void SetUp(const benchmark::State& state) override {
        SetUpImpl(state);
    }
    
    void TearDown(benchmark::State& state) override {
        TearDownImpl(state);
    }
    
    void TearDown(const benchmark::State& state) override {
        TearDownImpl(state);
    }

private:
    void SetUpImpl(const benchmark::State& state) {
        const auto size = static_cast<std::size_t>(state.range(0));
        const auto density = static_cast<std::size_t>(state.range(1));
        
        const auto nnz_per_row = size * density / 100;
        
        matrix_ = create_random_matrix(size, nnz_per_row);
        vector_ = create_random_vector(size);
        pool_ = std::make_unique<execution::ThreadPool>();
    }
    
    void TearDownImpl([[maybe_unused]] const benchmark::State& state) {
        matrix_ = SparseMatrix<T>{0, 0};
        vector_.clear();
        pool_.reset();
    }
    
    static SparseMatrix<T> create_random_matrix(std::size_t size, std::size_t nnz_per_row) {
        SparseMatrix<T> matrix(size, size);
        std::mt19937 gen(42);
        std::uniform_int_distribution<std::size_t> col_dist(0, size - 1);
        std::uniform_real_distribution<T> val_dist(1.0, 2.0);
        
        for (std::size_t i = 0; i < size; ++i) {
            std::vector<std::size_t> cols;
            cols.reserve(nnz_per_row);
            
            while (cols.size() < nnz_per_row) {
                std::size_t col = col_dist(gen);
                if (std::find(cols.begin(), cols.end(), col) == cols.end()) {
                    cols.push_back(col);
                    matrix.insert(i, col, val_dist(gen));
                }
            }
        }
        return matrix;
    }
    
    static std::vector<T> create_random_vector(std::size_t size) {
        std::vector<T> vec(size);
        std::mt19937 gen(42);
        std::uniform_real_distribution<T> val_dist(1.0, 2.0);
        std::generate(vec.begin(), vec.end(), [&]() { return val_dist(gen); });
        return vec;
    }
    
protected:    
    SparseMatrix<T> matrix_{0, 0};
    std::vector<T> vector_;
    std::unique_ptr<execution::ThreadPool> pool_;
};

} // anonymous namespace

BENCHMARK_TEMPLATE_DEFINE_F(BenchmarkFixture, Sequential, double)
(benchmark::State& state) {
    for (auto _ : state) {
        auto result = MatrixOps<double>::multiply(matrix_, vector_);
        benchmark::DoNotOptimize(result);
    }
    
    const auto iterations = static_cast<std::uint64_t>(state.iterations());
    const auto nnz = static_cast<std::uint64_t>(matrix_.nnz());
    state.SetItemsProcessed(static_cast<std::int64_t>(iterations * nnz));
    
    const auto bytes = iterations * nnz * 
        (static_cast<std::uint64_t>(sizeof(double)) + 
         static_cast<std::uint64_t>(sizeof(std::size_t)));
    state.SetBytesProcessed(static_cast<std::int64_t>(bytes));

    state.SetComplexityN(static_cast<benchmark::ComplexityN>(nnz));
}

BENCHMARK_TEMPLATE_DEFINE_F(BenchmarkFixture, Parallel, double)
(benchmark::State& state) {
    for (auto _ : state) {
        auto result = MatrixOps<double>::multiply_parallel(matrix_, vector_, *pool_);
        benchmark::DoNotOptimize(result);
    }
    
    const auto iterations = static_cast<std::uint64_t>(state.iterations());
    const auto nnz = static_cast<std::uint64_t>(matrix_.nnz());
    state.SetItemsProcessed(static_cast<std::int64_t>(iterations * nnz));
    
    const auto bytes = iterations * nnz * 
        (static_cast<std::uint64_t>(sizeof(double)) + 
         static_cast<std::uint64_t>(sizeof(std::size_t)));
    state.SetBytesProcessed(static_cast<std::int64_t>(bytes));

    state.SetComplexityN(static_cast<benchmark::ComplexityN>(nnz));
}

BENCHMARK_REGISTER_F(BenchmarkFixture, Sequential)
    ->Args({1000, 1})   // 1000x1000 matrix with 1% density
    ->Args({1000, 5})   // 1000x1000 matrix with 5% density
    ->Args({5000, 1})   // 5000x5000 matrix with 1% density
    ->Args({5000, 5})   // 5000x5000 matrix with 5% density
    ->Complexity(benchmark::oN)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(BenchmarkFixture, Parallel)
    ->Args({1000, 1})
    ->Args({1000, 5})
    ->Args({5000, 1})
    ->Args({5000, 5})
    ->Complexity(benchmark::oN)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

BENCHMARK_MAIN();
