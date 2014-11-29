#include "ParamServiceHandler.h"
#include "../gen-cpp/distrust_types.h"

namespace distrust {

void
ParamServiceHandler::announce(
  AnnounceResponse& _return,
  const int32_t worker_port) {

  // Open reverse connection
  std::cout << "Announce from " << worker_ip_ << std::endl;
  server_->add_worker(worker_ip_, worker_port);

  // Return parameters
  _return.model_info = server_->model_info_;
  _return.shard_paths = server_->shard_paths_;
  _return.learn_rate = 0.05;
  _return.batch_size = 128;
  server_->model_->get_params(_return.params);
}

void
ParamServiceHandler::push_update(const ParamUpdate &update) {
  printf("push_update\n");
}

void
ParamServiceHandler::pull_params(Params &_return) {
  printf("pull_params\n");
}

}  // namespace distrust

