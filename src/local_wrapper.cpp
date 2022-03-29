#include "spomp/local_wrapper.h"
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <cv_bridge/cv_bridge.h>

namespace spomp {

LocalWrapper::LocalWrapper(ros::NodeHandle& nh) : 
  nh_(nh), local_(createLocal(nh)), remote_(50) 
{
  obs_pano_viz_pub_ = nh_.advertise<sensor_msgs::Image>("obs_pano_viz", 1);
}

Local LocalWrapper::createLocal(ros::NodeHandle& nh) {
  return Local(TerrainPano::Params());
}

void LocalWrapper::play() {
  std::string bag_path;
  if (!nh_.getParam("bag_path", bag_path)) {
    std::cerr << "ERROR: No bag specified" << std::endl;
    return;
  }

  rosbag::Bag bag; 
  bag.open(bag_path, rosbag::bagmode::Read);

  for (const rosbag::MessageInstance& m : rosbag::View(bag)) {
    if (!ros::ok()) {
      break;
    }
    if (m.getTopic() == "/os_node/llol_odom/pano") {
      sensor_msgs::Image::ConstPtr pano = m.instantiate<sensor_msgs::Image>();
      if (pano != NULL) {
        if (!remote_.wait()) {
          // Want to quit
          break;
        }

        panoCallback(pano);
        visualizePano(pano->header.stamp);
      }
    }
    ros::spinOnce();
  }
}

void LocalWrapper::initialize() {
}

void LocalWrapper::panoCallback(const sensor_msgs::Image::ConstPtr& img_msg) {
  Eigen::MatrixXf pano_eig;
  cv_bridge::CvImageConstPtr pano_cv = cv_bridge::toCvShare(img_msg);
  cv::Mat depth_pano_cv(pano_cv->image.rows, pano_cv->image.cols, CV_16UC1);
  constexpr int from_to[] = {0, 0};
  // Extract first channel (depth)
  cv::mixChannels(&(pano_cv->image), 1, &depth_pano_cv, 1, from_to, 1);
  // Convert to Eigen and scale to meters
  cv::cv2eigen(depth_pano_cv, pano_eig);
  pano_eig /= 512;

  local_.updatePano(pano_eig, {});
}

void LocalWrapper::visualizePano(const ros::Time& stamp) {
  const Eigen::MatrixXi& pano = local_.getPano().getTraversability().matrix();
  cv::Mat pano_viz;
  cv::eigen2cv(pano, pano_viz);
  pano_viz.convertTo(pano_viz, CV_8UC1, 100);

  std_msgs::Header header;
  header.stamp = stamp;
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "mono8", pano_viz).toImageMsg();
  obs_pano_viz_pub_.publish(msg);
}

} // namespace spomp
