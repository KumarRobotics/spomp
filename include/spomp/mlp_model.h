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
      int hidden_layer_size = 10;
      float regularization = 0.01;
    };
    MLPModel(const Params& p);

    /*!
     * @param feat Set of feature vectors.  Each sample should be a column.
     * Since Eigen stores things column-major this makes accessing a sample
     * fast, since a sample is contiguous in memory.
     * @param labels Vector of labels
     */
    void fit(const Eigen::ArrayXXf& feat, const Eigen::VectorXi& labels);

    Eigen::VectorXf infer(const Eigen::ArrayXXf& feat);

    bool trained() const {
      return is_trained_;
    }

  private:
    /*********************************************************
     * LOCAL FUNCTIONS
     *********************************************************/
    arma::mat preprocFeatures(const Eigen::ArrayXXf& feat, bool update_stats = false);

    /*********************************************************
     * LOCAL CONSTANTS
     *********************************************************/
    Params params_;

    /*********************************************************
     * LOCAL VARIABLES
     *********************************************************/
    mlpack::FFN<mlpack::NegativeLogLikelihood, mlpack::RandomInitialization> model_;
    bool is_trained_{false};

    arma::colvec feat_means_;
    arma::colvec feat_stds_;
};

} // namespace spomp
