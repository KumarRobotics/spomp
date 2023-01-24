#include <MiniDNN/MiniDNN.h>
#include "spomp/aerial_map.h"

namespace spomp {
using namespace MiniDNN;

typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;

AerialMap::AerialMap(const Params& p) : params_(p) {}

void AerialMap::testML() {
  // Create two dimensional input data
  Vector x1 = Vector::LinSpaced(1000, 0.0, 3.15);
  Vector x2 = Vector::LinSpaced(1000, 0.0, 3.15);
  // Predictors -- each column is an observation
  Matrix x = Matrix::Random(2, 1000);
  x.row(0) = x1;
  x.row(1) = x2;
  // Response variables -- each column is an observation
  Matrix y = Matrix::Random(1, 1000);

  // Fill the output for the training
  for (int i = 0; i < y.cols(); i++)
  {
      y(0, i) = std::pow(x(0, i), 2) + std::pow(x(1, i), 2);
  }

  // Fill the output for the test
  Matrix xt = (Matrix::Random(2, 1000).array() + 1.0) / 2 * 3.15;
  Matrix yt = Matrix::Random(1, 1000);

  for (int i = 0; i < yt.cols(); i++)
  {
      yt(0, i) = std::pow(xt(0, i), 2) + std::pow(xt(1, i), 2);
  }

  // Construct a network object
  Network net;
  // Create three layers
  // Layer 1 -- FullyConnected, input size 2x200
  Layer* layer1 = new FullyConnected<Identity>(2, 100);
  // Layer 2 -- max FullyConnected, input size 200x200
  Layer* layer2 = new FullyConnected<ReLU>(100, 100);
  // Layer 4 -- fully connected, input size 200x1
  Layer* layer3 = new FullyConnected<Identity>(100, 1);
  // Add layers to the network object
  net.add_layer(layer1);
  net.add_layer(layer2);
  net.add_layer(layer3);
  // Set output layer
  net.set_output(new RegressionMSE());
  // Create optimizer object
  Adam opt;
  opt.m_lrate = 0.01;
  // (Optional) set callback function object
  //VerboseCallback callback;
  //net.set_callback(callback);
  // Initialize parameters with N(0, 0.01^2) using random seed 123
  net.init(0, 0.01, 000);
  Matrix pred = net.predict(xt);
  std::cout << (pred - yt).norm() << std::endl;
  // Fit the model with a batch size of 100, running 10 epochs with random seed 123
  net.fit(opt, x, y, 1000, 1000, 000);
  // Obtain prediction -- each column is an observation
  pred = net.predict(xt);
  std::cout << (pred - yt).norm() << std::endl;
}

} // namespace spomp
