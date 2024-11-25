#pragma once

#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <future>
#include <vector>
#include <stop_token>

namespace sparse_linalg::execution {

class ThreadPool {
private:
    class Worker {
    public:
        Worker(ThreadPool& pool) : pool_(pool) {}
        
        void operator()(std::stop_token stoken) {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock lock(pool_.mutex_);
                    pool_.cv_.wait(lock, [this, &stoken] {
                        return stoken.stop_requested() || !pool_.tasks_.empty();
                    });
                    
                    // Check if we should exit
                    if (stoken.stop_requested() && pool_.tasks_.empty()) {
                        return;
                    }
                    
                    if (!pool_.tasks_.empty()) {
                        task = std::move(pool_.tasks_.front());
                        pool_.tasks_.pop();
                        ++pool_.active_tasks_;
                    }
                }
                
                if (task) {
                    try {
                        task();
                    } catch (...) {
                        std::lock_guard<std::mutex> lock(pool_.exception_mutex_);
                        pool_.exceptions_.push_back(std::current_exception());
                    }
                    
                    {
                        std::lock_guard<std::mutex> lock(pool_.mutex_);
                        --pool_.active_tasks_;
                        if (pool_.active_tasks_ == 0 && pool_.tasks_.empty()) {
                            pool_.completed_cv_.notify_all();
                        }
                    }
                }
            }
        }
        
    private:
        ThreadPool& pool_;
    };

public:
    explicit ThreadPool(std::size_t num_threads = std::thread::hardware_concurrency()) {
        if (num_threads == 0) {
            throw std::invalid_argument("Thread pool must have at least one thread");
        }
        
        active_tasks_ = 0;
        
        try {
            for (std::size_t i = 0; i < num_threads; ++i) {
                workers_.emplace_back(Worker(*this));
            }
        } catch (...) {
            shutdown();
            throw;
        }
    }
    
    ~ThreadPool() {
        shutdown();
    }
    
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) {
        using return_type = std::invoke_result_t<F, Args...>;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        auto future = task->get_future();
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (workers_.empty()) {
                throw std::runtime_error("ThreadPool is shutting down");
            }
            
            tasks_.emplace([task]() { (*task)(); });
        }
        
        cv_.notify_one();
        return future;
    }
    
    void wait_all() {
        std::unique_lock lock(mutex_);
        completed_cv_.wait(lock, [this] { 
            return active_tasks_ == 0 && tasks_.empty();
        });
        
        // Check for exceptions
        std::lock_guard<std::mutex> ex_lock(exception_mutex_);
        if (!exceptions_.empty()) {
            std::rethrow_exception(exceptions_.front());
        }
    }
    
    [[nodiscard]] std::size_t thread_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return workers_.size();
    }

private:
    friend class Worker;
    
    void shutdown() {
        {
            std::unique_lock lock(mutex_);
            
            // Wait for all tasks to complete
            completed_cv_.wait(lock, [this] {
                return active_tasks_ == 0 && tasks_.empty();
            });
        }
        
        // Now stop the workers
        for (auto& worker : workers_) {
            worker.request_stop();
        }
        
        cv_.notify_all();
        
        workers_.clear();
    }
    
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable completed_cv_;
    std::queue<std::function<void()>> tasks_;
    std::vector<std::jthread> workers_;
    std::size_t active_tasks_;
    
    std::mutex exception_mutex_;
    std::vector<std::exception_ptr> exceptions_;
};

} // namespace sparse_linalg::execution