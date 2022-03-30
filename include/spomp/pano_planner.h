#pragma once

namespace spomp {

class PanoPlanner {
  public:
    struct Params {
    };

    PanoPlanner(const Params& params);

  private:
    Params params_;
};

} // namespace spomp
