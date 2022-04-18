#pragma once

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>

using gtsam::symbol_shorthand::P;

namespace spomp {

//! Wraps GTSAM.  Based of off ASOOM PoseGraph
class PoseGraph {
  public:
    struct Params {
      int num_frames_opt = 10;
    };
    PoseGraph(const Params& params);

    //! Add new node to graph
    size_t addNode(long stamp, const Eigen::Isometry3d& pose);

    //! Add prior to node.  Must have stamp matching exactly
    void addPrior(long stamp, const Eigen::Isometry3d& prior);

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
    Params params_;
    
    /***********************************************************
     * LOCAL VARIABLES
     ***********************************************************/
    //! GTSAM factor graph
    gtsam::NonlinearFactorGraph graph_;

    //! Keep track of the current best estimates for nodes
    gtsam::Values current_opt_;

    struct OriginalPose {
      Eigen::Isometry3d pose;
      gtsam::Key key;

      OriginalPose(const Eigen::Isometry3d& p, const gtsam::Key &k) : pose(p), key(k) {}
    };

    //! Map to keep track of timestamp to original poses
    std::map<long, OriginalPose> pose_history_;

    //! Number of frames in graph
    size_t size_;

    //! Number of frames in graph at time of last optimization
    size_t last_opt_size_;
};

} // namespace spomp
