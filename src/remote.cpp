#include "spomp/remote.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace spomp {

Remote::Remote(int wait_ms) : wait_ms_(wait_ms) {}

bool Remote::wait() {
  cv::Mat img = cv::Mat::zeros(50, 200, CV_8UC1);
  std::string text = std::to_string(counter_);
  if (counter_ < 0) {
    text = "press key";
  }
  cv::putText(img, text, cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX,
      1, cv::Scalar(255));
  cv::imshow("remote", img);

  int key_id;
  if (paused_) {
    key_id = cv::waitKey(-1);
  } else {
    key_id = cv::waitKey(wait_ms_);
  }

  key_id = key_id & 0xff;
  switch (key_id) {
    case ' ':
      paused_ = !paused_;
      break;
    case 'q':
      return false;
    default:
      break;
  }

  ++counter_;
  return true;
}

} // namespace spomp
