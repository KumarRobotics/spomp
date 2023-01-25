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

  // This does a copy, but we need to convert to double and normalize
  // So unavoidable
  arma::mat feat_arma_mat = arma::conv_to<arma::mat>::from(feat_arma.t());
  feat_means_ = arma::mean(feat_arma_mat, 1);
  feat_stds_ = arma::stddev(feat_arma_mat, /*normalize using N*/ 1, /*dim*/ 1);

  // Normalize
  feat_arma_mat.each_col() -= feat_means_;
  feat_arma_mat.each_col() /= feat_stds_;

  // Can only train using double matrices
  model_.Train(feat_arma_mat, 
      arma::conv_to<arma::mat>::from(labels_arma.t()));
}

Eigen::VectorXf MLPModel::infer(const Eigen::ArrayXXf& feat) {
  Eigen::VectorXf log_probs;
  if (feat_means_.n_elem > 0) {
    const arma::Mat<float> feat_arma(const_cast<float*>(feat.data()), 
        feat.rows(), feat.cols(), false, false);
    arma::mat feat_arma_mat = arma::conv_to<arma::mat>::from(feat_arma.t());

    // Normalize
    feat_arma_mat.each_col() -= feat_means_;
    feat_arma_mat.each_col() /= feat_stds_;

    arma::mat preds;
    model_.Predict(feat_arma_mat, preds);

    Eigen::Map<Eigen::ArrayXXd> preds_eig(preds.memptr(), preds.n_rows, preds.n_cols);
    // Row 1 is log(Prob(class 1)), which is nice since that way a prob near 1 
    // suggests class 1.
    log_probs = preds_eig.cast<float>().row(1);
  }

  return log_probs;
}

} // namespace spomp
