#pragma once

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>

namespace Eigen {
  using Vector6d = Matrix<double, 6, 1>;
}
using gtsam::symbol_shorthand::P;

namespace spomp {

//! Wraps GTSAM.  Based of off ASOOM PoseGraph
class PoseGraph {
  public:
    struct Params {
      int num_frames_opt = 10;
      Eigen::Vector6d between_uncertainty = Eigen::Vector6d::Constant(0.1);
      Eigen::Vector3d prior_uncertainty = Eigen::Vector3d::Constant(0.1);
      bool allow_interpolation = false;

      void setBetweenUncertainty(double loc, double rot) {
        // gtsam stores rot info first
        between_uncertainty.head<3>().setConstant(rot);
        between_uncertainty.tail<3>().setConstant(loc);
      }

      void setPriorUncertainty(double loc, double rot) {
        // gtsam stores rot info first
        prior_uncertainty.head<1>().setConstant(rot);
        prior_uncertainty.tail<2>().setConstant(loc);
      }
    };
    PoseGraph(const Params& params);

    //! Add new node to graph
    size_t addNode(long stamp, const Eigen::Isometry3d& pose);

    struct Prior2D {
      Eigen::Isometry2d pose = Eigen::Isometry2d::Identity();
      Eigen::Vector3d sigma_diag = Eigen::Vector3d::Constant(-1);
      // Pose in odom frame at time of prior
      // Leave identity if not available
      Eigen::Isometry3d local_pose = Eigen::Isometry3d::Identity();

      // Constructor for not specifying sigma
      Prior2D(const Eigen::Isometry2d& p) : pose(p) {}
      Prior2D() = default;
    };
    //! Add prior to node.  Must have stamp matching exactly
    void addPrior(long stamp, const Prior2D& prior);

    //! Run gtsam optimization
    void update();

    /*!
     * Get pose of particular node in graph
     *
     * @param stamp Timestamp in nsec to get pose of
     * @return Pose of node.  If no node at timestamp, return nothing
     */
    std::optional<Eigen::Isometry3d> getPoseAtTime(long stamp) const;

    /*!
     * Get pose of particular node in graph
     *
     * @param ind Index of node to get pose of
     * @return Pose of node.
     */
    Eigen::Isometry3d getPoseAtIndex(size_t ind) const;


    //! @return Number of nodes in graph
    size_t size() const {
      return size_;
    }

    //! @return Current error in graph
    double getError() const {
      return graph_.error(current_opt_);
    }

  private:
    /***********************************************************
     * LOCAL FUNCTIONS
     ***********************************************************/
    //! Add global priors to graph off of buffer
    void processGlobalBuffer();

    void addPriorFactor(const gtsam::Key& key, const Prior2D& prior);

    //! Convert GTSAM pose to Eigen
    static gtsam::Pose3 Eigen2GTSAM(const Eigen::Isometry3d& eigen_pose) {
      return gtsam::Pose3(eigen_pose.matrix());
    }

    //! Convert Eigen pose to GTSAM
    static Eigen::Isometry3d GTSAM2Eigen(const gtsam::Pose3& gtsam_pose) {
      return Eigen::Isometry3d(gtsam_pose.matrix());
    }

    /***********************************************************
     * LOCAL CONSTANTS
     ***********************************************************/
    Params params_{};
    
    /***********************************************************
     * LOCAL VARIABLES
     ***********************************************************/
    //! GTSAM factor graph
    gtsam::NonlinearFactorGraph graph_{};

    //! Keep track of the current best estimates for nodes
    gtsam::Values current_opt_{};

    struct OriginalPose {
      Eigen::Isometry3d pose;
      gtsam::Key key;
    };

    //! Map to keep track of timestamp to original poses
    std::map<long, OriginalPose> pose_history_{};

    //! Buffer of priors
    std::map<long, Prior2D> prior_buffer_{};

    //! Number of frames in graph
    size_t size_{0};

    //! Number of frames in graph at time of last optimization
    size_t last_opt_size_{0};

    //! Index of temporary origin factor before adding priors
    int initial_pose_factor_id_{};
};

} // namespace spomp
