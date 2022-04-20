#pragma once

#include <chrono>
#include <list>
#include <string>
#include <ostream>
#include <cmath>
#include <mutex>

namespace spomp {

/*!
 * Simple Timer class to stop, start, and generate stats
 */
class Timer {
  public:
    Timer(const std::string& name, bool mt = false) : 
      name_(name), multithread_(mt) {}
    Timer() = default;

    void start() {
      // Don't lock here, since we assume a given timer
      // is only called from one thread
      start_t_ = std::chrono::system_clock::now();
    }

    void end();

    friend std::ostream& operator<<(std::ostream& os, Timer& t);

  private:
    // These are private since they are not thread-safe
    double avg_us() const;

    double std_us() const;

    double last_us() const {
      return last_t_;
    }

    int count() const {
      return n_;
    }

    std::string name_{};

    // Threading stuff
    bool multithread_{false};
    std::mutex mtx_{};

    // These things integrate us (or squared us), so make big
    long long t_sum_{0};
    long long t_sq_sum_{0};
    long last_t_{0};
    int n_{0};
    std::chrono::time_point<std::chrono::system_clock> start_t_{};
};

class TimerManager {
  public:
    //! Return global TimerManager instance
    static TimerManager& getGlobal(bool multithread = false) {
      static TimerManager tm(multithread);
      return tm;
    }

    TimerManager(bool mt = false) : multithread_(mt) {}

    Timer* get(const std::string& name) {
      std::unique_lock lock(mtx_, std::defer_lock);
      if (multithread_) lock.lock();

      return &timers_.emplace_back(name, multithread_);
    }

    friend std::ostream& operator<<(std::ostream& os, TimerManager& tm);

  private:
    bool multithread_{false};
    std::mutex mtx_{};

    //! We use a list because lists are reference-safe
    //! For a vector, invalidated on realloc
    std::list<Timer> timers_{};
};

} // namespace spomp
