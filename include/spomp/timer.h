#pragma once

#include <chrono>
#include <list>
#include <string>
#include <ostream>
#include <cmath>

namespace spomp {

/*!
 * Simple Timer class to stop, start, and generate stats
 */
class Timer {
  public:
    Timer(const std::string& name) : name_(name) {}
    Timer() = default;

    void start() {
      start_t_ = std::chrono::system_clock::now();
    }

    void end();

    double avg_us() const;

    double std_us() const;

    double last_us() const {
      return last_t_;
    }

    int count() const {
      return n_;
    }

    friend std::ostream& operator<<(std::ostream& os, const Timer& t);

  private:
    std::string name_{};
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
    static TimerManager& getGlobal() {
      static TimerManager tm{};
      return tm;
    }

    TimerManager() = default;

    Timer* get(const std::string& name) {
      return &timers_.emplace_back(name);
    }

    friend std::ostream& operator<<(std::ostream& os, const TimerManager& tm);

  private:
    //! We use a list because lists are reference-safe
    //! For a vector, invalidated on realloc
    std::list<Timer> timers_{};
};

} // namespace spomp
