#include "ParamServiceHandler.h"

#include <iostream>

#include "../gen-cpp/distrust_types.h"

namespace distrust {

namespace {

std::string
get_worker_key(const std::string &ip, const int32_t port) {
  std::stringstream ss;
  ss << ip << port;
  return ss.str();
}

}  // namespace

void
ParamServiceHandler::announce(
  AnnounceResponse& _return,
  const int32_t worker_port) {

  // Open reverse connection
  std::cout << "announce from " << worker_ip_ << ":" << worker_port
            << std::endl;
  server_->add_worker(worker_ip_, worker_port);

  // Return parameters
  _return.model_info = server_->model_info_;
  std::unique_lock<std::mutex> shard_paths_write_lock(
      server_->shard_paths_lock_);
  _return.shard_paths = server_->worker_to_shards_[
      get_worker_key(worker_ip_, worker_port)];
  shard_paths_write_lock.unlock();
  _return.learn_rate = 0.025;
  _return.batch_size = 128;
  std::unique_lock<std::mutex> model_read_lock(
      server_->model_lock_);
  server_->model_->get_params(_return.params);
  model_read_lock.unlock();
}

void
ParamServiceHandler::push_update(const ParamUpdate &update) {
  //std::cout << "push_update" << std::endl;
  std::unique_lock<std::mutex> model_write_lock(
      server_->model_lock_);
  server_->model_->update_params(update);
  model_write_lock.unlock();
}

void
ParamServiceHandler::pull_params(Params &_return) {
  //std::cout << "pull_params" << std::endl;
  std::unique_lock<std::mutex> model_read_lock(
      server_->model_lock_);
  server_->model_->get_params(_return);
  model_read_lock.unlock();
}

}  // namespace distrust

