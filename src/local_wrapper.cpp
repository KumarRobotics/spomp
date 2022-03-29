#include "spomp/local_wrapper.h"
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

namespace spomp {

LocalWrapper::LocalWrapper(ros::NodeHandle& nh) : 
  nh_(nh), local_(createLocal(nh)), remote_(50) 
{
  auto& tm = TimerManager::getGlobal();
  viz_pano_t_ = tm.get("LW_viz_pano");
  viz_cloud_t_ = tm.get("LW_viz_cloud");

  obs_pano_viz_pub_ = nh_.advertise<sensor_msgs::Image>("obs_pano_viz", 1);
  obs_cloud_viz_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("obs_cloud_viz", 1);
}

Local LocalWrapper::createLocal(ros::NodeHandle& nh) {
  TerrainPano::Params params{};
  nh.getParam("tbb", params.tbb);

  return Local(params);
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
        visualizeCloud(pano->header.stamp);
        printTimings();
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
  viz_pano_t_->start();

  const Eigen::MatrixXi& pano = local_.getPano().getTraversability().matrix();
  cv::Mat pano_viz;
  cv::eigen2cv(pano, pano_viz);
  pano_viz.convertTo(pano_viz, CV_8UC1, 100);

  std_msgs::Header header;
  header.stamp = stamp;
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "mono8", pano_viz).toImageMsg();
  obs_pano_viz_pub_.publish(msg);

  viz_pano_t_->end();
}

void LocalWrapper::visualizeCloud(const ros::Time& stamp) {
  viz_cloud_t_->start();

  // Grab data
  const auto& cloud = local_.getPano().getCloud();
  const auto& trav = local_.getPano().getTraversability();

  sensor_msgs::PointCloud2 cloud_msg;
  cloud_msg.header.stamp = stamp;
  cloud_msg.header.frame_id = "map";
  cloud_msg.height = 1;
  cloud_msg.width = cloud[0].size();

  {
    sensor_msgs::PointField field;
    field.name = "x";
    field.offset = 0;
    field.datatype = sensor_msgs::PointField::FLOAT32;
    field.count = 1;
    cloud_msg.fields.push_back(field);
  }
  {
    sensor_msgs::PointField field;
    field.name = "y";
    field.offset = 4;
    field.datatype = sensor_msgs::PointField::FLOAT32;
    field.count = 1;
    cloud_msg.fields.push_back(field);
  }
  {
    sensor_msgs::PointField field;
    field.name = "z";
    field.offset = 8;
    field.datatype = sensor_msgs::PointField::FLOAT32;
    field.count = 1;
    cloud_msg.fields.push_back(field);
  }
  {
    sensor_msgs::PointField field;
    field.name = "data";
    field.offset = 12;
    field.datatype = sensor_msgs::PointField::FLOAT32;
    field.count = 1;
    cloud_msg.fields.push_back(field);
  }

  cloud_msg.point_step = 16;
  cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;
  cloud_msg.is_dense = true;

  cloud_msg.data.resize(cloud_msg.row_step*cloud_msg.height);

  // Map eigen array onto pc data
  Eigen::Map<Eigen::ArrayXXf> cloud_packed(reinterpret_cast<float*>(cloud_msg.data.data()), 
      4, cloud[0].size());

  // Copy data over
  for (int axis=0; axis<3; ++axis) {
    cloud_packed.row(axis) = Eigen::Map<const Eigen::VectorXf>(
        cloud[axis].data(), cloud[axis].size());
  }
  cloud_packed.row(3) = Eigen::Map<const Eigen::VectorXi>(
      trav.data(), trav.size()).cast<float>();

  obs_cloud_viz_pub_.publish(cloud_msg);

  viz_cloud_t_->end();
}

void LocalWrapper::printTimings() {
  ROS_INFO_STREAM("\033[34m" << TimerManager::getGlobal() << "\033[0m");
}

} // namespace spomp
