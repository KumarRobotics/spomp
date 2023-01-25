#include "spomp/mlp_model.h"

namespace spomp {

MLPModel::MLPModel(const Params& p) : params_(p) {
  model_.Add<mlpack::LinearType<arma::mat, mlpack::L2Regularizer>>(
      params_.hidden_layer_size, params_.regularization);
  model_.Add<mlpack::Sigmoid>();
  model_.Add<mlpack::LinearType<arma::mat, mlpack::L2Regularizer>>(
      2, params_.regularization);
  model_.Add<mlpack::LogSoftMax>();
}

void MLPModel::fit(const Eigen::ArrayXXf& feat, 
    const Eigen::VectorXi& labels) 
{
  // Do not want to copy data
  // Somewhat sketchy, since armadillo doesn't have a non-copy constructor
  // for const data.  We pinky promise to not modify this.
  const arma::Mat<float> feat_arma(const_cast<float*>(feat.data()), 
      feat.rows(), feat.cols(), false, false);
  const arma::Mat<int> labels_arma(const_cast<int*>(labels.data()), 
      labels.rows(), labels.cols(), false, false);

  // Can only train using double matrices
  model_.Train(arma::conv_to<arma::mat>::from(feat_arma.t()), 
      arma::conv_to<arma::mat>::from(labels_arma.t()));
}

Eigen::VectorXf MLPModel::infer(const Eigen::ArrayXXf& feat) {
  const arma::Mat<float> feat_arma(const_cast<float*>(feat.data()), 
      feat.rows(), feat.cols(), false, false);

  arma::mat preds;
  model_.Predict(arma::conv_to<arma::mat>::from(feat_arma.t()), preds);

  Eigen::VectorXf log_probs;
  Eigen::Map<Eigen::ArrayXXd> preds_eig(preds.memptr(), preds.n_rows, preds.n_cols);
  log_probs = preds_eig.cast<float>().row(1);

  return log_probs;
}

} // namespace spomp
