#ifndef WORKER_H
#define WORKER_H

#include <pthread.h>

#include "../gen-cpp/ParamService.h"
#include "LanguageModel.h"

class Worker {
public:
  Worker(const std::string& master_ip, const int master_port, const int worker_port);
  ~Worker() {}
  void run();

protected:
  static void *announce(void *arg);
  static void *server(void *arg);

protected:
  std::unique_ptr<distrust::ParamServiceClient> param_client_;
  std::unique_ptr<LanguageModel> model_;

  // Paths to training data
  std::vector<std::string> shard_paths_;

  // Learning rate
  double learn_rate_;

  pthread_t server_thread_, announce_thread_, pull_thread_, push_thread_, compute_thread_;
  std::string master_ip_;
  int master_port_, worker_port_;

  // server/client synchronization
  pthread_mutex_t lock_;
  bool server_ready_;
  pthread_cond_t server_ready_cond_;
};

#endif
