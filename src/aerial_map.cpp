#include "spomp/aerial_map.h"

namespace spomp {

AerialMap::AerialMap(const Params& p) : params_(p) {}

void AerialMap::testML() {
    using namespace dlib;
    using net_type = loss_multiclass_log<
      fc<2,
      sig<fc<100,
      input<matrix<float>>
      >>>>;

    using test_net_type = softmax<
      fc<2,
      sig<fc<100,
      input<matrix<float>>
      >>>>;

    net_type train_net;
    dnn_trainer<net_type, adam> trainer(train_net, adam(0.0005, 0.9, 0.999));
    trainer.be_verbose();

    std::vector<matrix<float>> train_inputs;
    std::vector<unsigned long> train_outputs;

    train_inputs.push_back(matrix<float, 4, 1>{1,1,1,1});
    train_outputs.push_back(0);
    train_inputs.push_back(matrix<float, 4, 1>{2,2,2,2});
    train_outputs.push_back(1);

    trainer.train(train_inputs, train_outputs);
}

} // namespace spomp
