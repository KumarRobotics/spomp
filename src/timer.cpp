#include "spomp/timer.h"
#include <array>

namespace spomp {

void Timer::end() {
  using namespace std::chrono;
  auto end_t = system_clock::now();
  last_t_ = duration_cast<microseconds>(end_t - start_t_).count();
  t_ += last_t_;
  n_++;
}

std::ostream& operator<<(std::ostream& os, const Timer& t) {
  static const std::array<std::string, 3> units = {"us", "ms", "s"};
  double mean = t.avg_us();
  double last = t.last_us();
  int unit_i = 0;

  while (mean > 100 && unit_i < units.size()-1) {
    mean /= 1000;
    last /= 1000;
    ++unit_i;
  }

  std::string unit = units[unit_i];
  os << "[" << t.name_ << "] mean: " << mean << " " << unit << 
        " last: " << last << " " << unit <<
        " count: " << t.count();
  return os;
}

std::ostream& operator<<(std::ostream& os, const TimerManager& tm) {
  for (const auto& timer : tm.timers_) {
    os << std::endl << timer;
  }
  return os;
}

} // namespace spomp
