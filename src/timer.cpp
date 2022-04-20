#include <array>
#include <iomanip>
#include "spomp/timer.h"

namespace spomp {

void Timer::end() {
  using namespace std::chrono;
  auto end_t = system_clock::now();

  std::unique_lock lock(mtx_, std::defer_lock);
  if (multithread_) lock.lock();
  last_t_ = duration_cast<microseconds>(end_t - start_t_).count();
  t_sum_ += last_t_;
  t_sq_sum_ += std::pow<long long>(last_t_, 2);
  ++n_;
}

double Timer::avg_us() const {
  if (n_ == 0) {
    return 0;
  }
  return t_sum_/static_cast<double>(n_);
}

double Timer::std_us() const {
  if (n_ < 2) {
    return 0;
  }
  return std::sqrt((t_sq_sum_/n_ - std::pow(avg_us(), 2)) * (n_ / (n_ - 1.)));
}

std::ostream& operator<<(std::ostream& os, Timer& t) {
  static const std::array<std::string, 3> units = {"us", "ms", "s"};
  double mean, last, std;
  int count;
  {
    std::unique_lock lock(t.mtx_, std::defer_lock);
    if (t.multithread_) lock.lock();
    mean = t.avg_us();
    last = t.last_us();
    std = t.std_us();
    count = t.count();
  }
  int unit_i = 0;

  while (mean > 100 && unit_i < units.size()-1) {
    mean /= 1000;
    last /= 1000;
    std /= 1000;
    ++unit_i;
  }

  std::string unit = units[unit_i];
  using namespace std;
  os << "[" << left << setw(20) << t.name_ << "]" << 
        " mean: " << setw(10) << mean << unit << 
        " last: " << setw(10) << last << unit <<
        " std: " << setw(10) << std << unit <<
        " count: " << setw(10) << count;
  return os;
}

std::ostream& operator<<(std::ostream& os, TimerManager& tm) {
  for (auto& timer : tm.timers_) {
    os << std::endl << timer;
  }
  return os;
}

} // namespace spomp
