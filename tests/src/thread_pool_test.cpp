#include <doctest/doctest.h>
#include <sparse_linalg/execution/thread_pool.hpp>
#include <atomic>
#include <chrono>
#include <vector>
#include <numeric>

using namespace sparse_linalg::execution;
using namespace std::chrono_literals;

TEST_SUITE("ThreadPool") {
    TEST_CASE("basic functionality") {
        ThreadPool pool(4);
        
        SUBCASE("simple task execution") {
            auto future = pool.submit([]() { return 42; });
            CHECK(future.get() == 42);
        }
        
        SUBCASE("multiple tasks") {
            constexpr std::size_t num_tasks = 100;
            std::vector<std::future<std::size_t>> futures;
            futures.reserve(num_tasks);
            
            for (std::size_t i = 0; i < num_tasks; ++i) {
                futures.push_back(pool.submit([i]() { return i; }));
            }
            
            for (std::size_t i = 0; i < futures.size(); ++i) {
                CHECK(futures[i].get() == i);
            }
            
            pool.wait_all();
        }
        
        SUBCASE("exception handling") {
            auto future = pool.submit([]() -> int { 
                throw std::runtime_error("test error"); 
            });
            
            CHECK_THROWS_AS(future.get(), std::runtime_error);
            
            // Pool should still be usable after exception
            auto future2 = pool.submit([]() { return 42; });
            CHECK(future2.get() == 42);
        }
    }
    
    TEST_CASE("stress testing") {
        ThreadPool pool(8);
        
        SUBCASE("concurrent increment") {
            std::atomic<std::size_t> counter{0};
            constexpr std::size_t num_increments = 10000;
            std::vector<std::future<void>> futures;
            futures.reserve(num_increments);
            
            for (std::size_t i = 0; i < num_increments; ++i) {
                futures.push_back(pool.submit([&counter]() {
                    counter.fetch_add(1, std::memory_order_relaxed);
                }));
            }
            
            pool.wait_all();
            CHECK(counter == num_increments);
        }
        
        SUBCASE("task timing") {
            constexpr std::size_t num_tasks = 100;
            auto start = std::chrono::steady_clock::now();
            
            std::vector<std::future<void>> futures;
            futures.reserve(num_tasks);
            
            for (std::size_t i = 0; i < num_tasks; ++i) {
                futures.push_back(pool.submit([]() {
                    std::this_thread::sleep_for(10ms);
                }));
            }
            
            pool.wait_all();
            
            auto duration = std::chrono::steady_clock::now() - start;
            //  100 tasks of 10ms each split to 8 threads = ~125ms
            CHECK(duration < std::chrono::milliseconds(200));
        }
    }
    
    TEST_CASE("shutdown behavior") {
        std::atomic<bool> task_completed{false};
        
        {
            ThreadPool pool(4);
            pool.submit([&task_completed]() {
                std::this_thread::sleep_for(50ms);
                task_completed = true;
            });
            // Pool destructor will be called here
        }
        
        CHECK(task_completed);
        
        SUBCASE("multiple tasks completion") {
            constexpr std::size_t num_tasks = 50;
            std::atomic<std::size_t> completed_tasks{0};
            
            {
                ThreadPool pool(4);
                for (std::size_t i = 0; i < num_tasks; ++i) {
                    pool.submit([&completed_tasks]() {
                        std::this_thread::sleep_for(10ms);
                        completed_tasks.fetch_add(1, std::memory_order_relaxed);
                    });
                }
            }
            
            CHECK(completed_tasks == num_tasks);
        }
    }
}