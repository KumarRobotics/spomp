#include <array>
#include <iomanip>
#include "spomp/timer.h"

namespace spomp {

/**
 * @brief Ends the timer and calculates the time duration.
 *
 * This function computes the time elapsed since the timer was started
 * and updates the necessary variables for statistics calculation.
 *
 * @note This function is thread-safe if the Timer object is constructed with multithreading enabled.
 *
 */
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

/**
 * @brief Calculates the average time in microseconds.
 *
 * This function calculates the average time in microseconds by dividing the total time sum, t_sum_,
 * by the number of measurements, n_. If there are no measurements (n_ = 0), the function returns 0.
 *
 * @return The average time in microseconds.
 */
    double Timer::avg_us() const {
  if (n_ == 0) {
    return 0;
  }
  return t_sum_/static_cast<double>(n_);
}

/**
 * @brief Calculates the standard deviation of the time measurements in microseconds.
 *
 * This function calculates the standard deviation of the time measurements in microseconds using
 * the formula: sqrt((t_sq_sum_/n_ - avg_us()^2) * (n_ / (n_ - 1))), where t_sq_sum_ is the sum of the squares
 * of the time measurements and n_ is the number of measurements.
 * If the number of measurements is less than 2, the function returns 0.
 *
 * @return The standard deviation of the time measurements in microseconds.
 */
    double Timer::std_us() const {
  if (n_ < 2) {
    return 0;
  }
  return std::sqrt((static_cast<double>(t_sq_sum_)/n_ - std::pow(avg_us(), 2)) * 
         (static_cast<double>(n_) / (n_ - 1.)));
}

/**
 * @brief Overloaded stream insertion operator for the Timer class
 *
 * This function is an overloaded insertion operator that allows a Timer object to
 * be inserted into an output stream. It prints the name, mean, last, standard deviation,
 * and count of the Timer object to the output stream.
 *
 * @param os The output stream to insert the Timer object into
 * @param t The Timer object to insert into the output stream
 * @return A reference to the output stream
 */
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

/**
 * @brief Overload of the output stream insertion operator for TimerManager objects.
 *
 * This function inserts the string representation of each Timer in the TimerManager into
 * the output stream, followed by a newline character.
 *
 * @param os The output stream to insert the TimerManager into.
 * @param tm The TimerManager to be inserted into the output stream.
 * @return The output stream with the TimerManager inserted.
 */
    std::ostream& operator<<(std::ostream& os, TimerManager& tm) {
  for (auto& timer : tm.timers_) {
    os << timer << std::endl;
  }
  return os;
}

} // namespace spomp
