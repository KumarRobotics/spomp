#pragma once

#include <chrono>
#include <list>
#include <string>
#include <iostream>

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

    long avg_us() const {
      if (n_ == 0) {
        return 0;
      }
      return t_/n_;
    }

    double last_us() const {
      return last_t_;
    }

    int count() const {
      return n_;
    }

    friend std::ostream& operator<<(std::ostream& os, const Timer& t);

  private:
    std::string name_{};
    long t_{0};
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
