#pragma once

#include <mlpack.hpp>
#include <Eigen/Dense>

namespace spomp {

/*! 
 * Basically act as a wrapper around mlpack ann.
 * Also useful to wrap the armadillo stuff and expose an Eigen interface
 */
class MLPModel {
  public:
    struct Params {
      int hidden_layer_size = 100;
      float regularization = 0.01;
    };
    MLPModel(const Params& p);

    void fit(const Eigen::ArrayXXf& feat, const Eigen::VectorXi& labels);

    Eigen::VectorXf infer(const Eigen::ArrayXXf& feat);

  private:
    /*********************************************************
     * LOCAL CONSTANTS
     *********************************************************/
    Params params_;

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    mlpack::FFN<mlpack::NegativeLogLikelihood, mlpack::RandomInitialization> model_;
};

} // namespace spomp
