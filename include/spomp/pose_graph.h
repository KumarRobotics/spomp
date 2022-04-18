#pragma once

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>

using gtsam::symbol_shorthand::P;

namespace spomp {

class PoseGraph {
  public:
    struct Params {
    };
    PoseGraph(const Params& params);

    //! Add new node to graph
    size_t addNode(long stamp, const Eigen::Isometry3f& pose);

    //! Add prior to node.  Must have stamp matching exactly
    void addPrior(long stamp, const Eigen::Isometry3f& prior);

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
    size_t size() const;

    //! @return Current error in graph
    size_t getError() const;

  private:
    /***********************************************************
     * LOCAL FUNCTIONS
     ***********************************************************/
    //! Add global priors to graph off of buffer
    void processGlobalBuffer();

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
};

} // namespace spomp
