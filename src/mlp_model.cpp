#include "spomp/mlp_model.h"

namespace spomp {

/**
 * @class MLPModel
 * @brief A class that represents a multi-layer perceptron (MLP) model.
 *
 * The MLPModel class is used to create and manage a MLP model for classification
 * tasks. It consists of multiple layers of linear transform, activation, and softmax
 * functions. The model can be trained using mlpack optimizer classes like
 * SGD (Stochastic Gradient Descent) or Adam.
 */
    MLPModel::MLPModel(const Params& p) : params_(p) {
  model_.Add<mlpack::LinearType<arma::mat, mlpack::L2Regularizer>>(
      params_.hidden_layer_size, params_.regularization);
  model_.Add<mlpack::Sigmoid>();
  model_.Add<mlpack::LinearType<arma::mat, mlpack::L2Regularizer>>(
      2, params_.regularization);
  model_.Add<mlpack::LogSoftMax>();
}

/**
 * Preprocesses the features by converting them to an Armadillo matrix, normalizing them, and update the statistics of mean and standard deviation.
 * @param feat The input Eigen array of features
 * @param update_stats A boolean flag indicating whether to update the statistics of mean and standard deviation
 * @return The preprocessed features as an Armadillo matrix
 */
    arma::mat MLPModel::preprocFeatures(const Eigen::ArrayXXf& feat, bool update_stats) {
  // Do not want to copy data
  // Somewhat sketchy, since armadillo doesn't have a non-copy constructor
  // for const data.  We pinky promise to not modify this.
  const arma::Mat<float> feat_arma(const_cast<float*>(feat.data()), 
      feat.rows(), feat.cols(), false, false);
  // This does a copy, but we need to convert to double and normalize
  // So unavoidable
  arma::mat feat_arma_mat = arma::conv_to<arma::mat>::from(feat_arma);
  if (update_stats) {
    feat_means_ = arma::mean(feat_arma_mat, 1);
    feat_stds_ = arma::stddev(feat_arma_mat, /*normalize using N*/ 1, /*dim*/ 1);
  }
  // Replace all zeros with 1 to avoid divide by 0
  feat_stds_.transform( [](double val) { return val == 0 ? 1 : val; } );

  // Normalize
  feat_arma_mat.each_col() -= feat_means_;
  feat_arma_mat.each_col() /= feat_stds_;

  return feat_arma_mat;
}

/**
 * @brief Fits the MLPModel using the given features and labels.
 *
 * This function trains the MLPModel with the provided features and labels.
 * It checks if the dimensions of the input data are valid and transforms the labels to an Armadillo matrix.
 * The features are preprocessed by setting them to be of double type.
 *
 * @param feat The input feature matrix represented as an Eigen::ArrayXXf.
 * @param labels The labels vector represented as an Eigen::VectorXi.
 */
    void MLPModel::fit(const Eigen::ArrayXXf& feat,
                       const Eigen::VectorXi& labels)
{
  if (feat.cols() == 0 || labels.size() == 0) return;
  const arma::Mat<int> labels_arma(const_cast<int*>(labels.data()), 
      labels.rows(), labels.cols(), false, false);

  // Can only train using double matrices
  model_.Train(preprocFeatures(feat, true), 
      arma::conv_to<arma::mat>::from(labels_arma.t()));

  is_trained_ = true;
}

/**
 * @brief Performs inference using a Multi-Layer Perceptron (MLP) model
 *
 * Given a feature matrix `feat`, this function predicts the probabilities
 * of the input belonging to class 1 using the trained MLP model.
 *
 * @param feat The feature matrix, of size (num_samples, num_features), where
 *             each row represents a sample and each column represents a feature.
 * @return An Eigen::VectorXf containing the log probabilities of the input
 *         belonging to class 1. If `feat_means_` is empty, the returned
 *         Eigen::VectorXf will also be empty.
 */
    Eigen::VectorXf MLPModel::infer(const Eigen::ArrayXXf& feat) {
  Eigen::VectorXf log_probs;
  if (feat_means_.n_elem > 0) {
    arma::mat preds;
    model_.Predict(preprocFeatures(feat), preds);

    Eigen::Map<Eigen::ArrayXXd> preds_eig(preds.memptr(), preds.n_rows, preds.n_cols);
    // Row 1 is log(Prob(class 1)), which is nice since that way a prob near 1 
    // suggests class 1.
    log_probs = preds_eig.cast<float>().row(1);
  }

  return log_probs;
}

} // namespace spomp
