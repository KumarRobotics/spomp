#pragma once

namespace spomp {

/*!
 * Simple class to manage pausing/starting/stopping data
 * playack.  Inspired by KeyControl from within
 * https://github.com/versatran01/dsol
 */
class Remote {
  public:
    Remote(int wait_ms);

    //! Wait, returns false to quit
    bool wait();

  private:
    const int wait_ms_;

    int counter_{-1};
    
    bool paused_{true};
};
  
} // namespace spomp
