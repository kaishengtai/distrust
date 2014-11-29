#include "Worker.h"

#include <iostream>
#include <pthread.h>

using boost::shared_ptr;

using namespace distrust;

void
WorkerServiceHandler::heartbeat(HBResponse &_return) {
  std::cout << "[heartbeat] request received" << std::endl;
  pthread_mutex_lock(&worker_->completed_shards_lock_);
  for (auto itr = worker_->completed_shards_.begin();
       itr != worker_->completed_shards_.end();
       itr++) {
    _return.completed_shards.push_back(*itr);
  }
  pthread_mutex_unlock(&worker_->completed_shards_lock_);
}

void
WorkerServiceHandler::start(const StartRequest &request) {
  std::cout << "[start] request received" << std::endl;
  pthread_mutex_lock(&worker_->shard_paths_lock_);
  pthread_mutex_lock(&worker_->stop_lock_);
  if (worker_->stop_) {
    worker_->stop_ = false;
    for (const std::string &path : request.shard_paths) {
      worker_->shard_paths_.push(path);
    }
    worker_->learn_rate_ = request.learn_rate;
    worker_->batch_size_ = request.batch_size;
    pthread_cond_signal(&worker_->stop_cond_);
    pthread_mutex_unlock(&worker_->stop_lock_);
    pthread_mutex_unlock(&worker_->shard_paths_lock_);
    std::cout << "[start] starting worker" << std::endl;
    std::cout << "[start] assigned shards:" << std::endl;
    for (const std::string &path : request.shard_paths) {
      std::cout << "\t" << path << std::endl;
    }
  } else {
    pthread_mutex_unlock(&worker_->stop_lock_);
    pthread_mutex_unlock(&worker_->shard_paths_lock_);
    std::cout << "[start] already started -- ignoring request" << std::endl;
  }
}

void
WorkerServiceHandler::stop() {
  std::cout << "[stop] request received" << std::endl;
  pthread_mutex_lock(&worker_->stop_lock_);
  worker_->stop_ = true;
  pthread_mutex_unlock(&worker_->stop_lock_);
}

void
WorkerServiceHandler::reassign(const std::vector<std::string> &shard_paths) {
  std::cout << "[reassign] request received" << std::endl;
  
  pthread_mutex_lock(&worker_->shard_paths_lock_);
  while (!worker_->shard_paths_.empty()) {
    worker_->shard_paths_.pop();
  }

  for (const std::string &path : shard_paths) {
    if (worker_->completed_shards_.find(path) == worker_->completed_shards_.end()) {
      worker_->shard_paths_.push(path);
    }
  }
  pthread_mutex_unlock(&worker_->shard_paths_lock_);

  std::cout << "[reassign] assigned shards:" << std::endl;
  for (const std::string &path : shard_paths) {
    std::cout << "\t" << path << std::endl;
  }
}
