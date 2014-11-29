#ifndef WORKER_H
#define WORKER_H

#include <pthread.h>
#include <queue>
#include <unordered_set>

#include "../gen-cpp/WorkerService.h"
#include "../gen-cpp/ParamService.h"
#include "../LanguageModel.h"

#define WORKER_DEFAULT_PORT 9090

class Worker {

 friend class WorkerServiceHandler;

 public:
  Worker(const std::string& master_ip,
         const int master_port,
         const int worker_port);
  ~Worker() {}
  void run();

 protected:
  static void *announce(void *arg);
  static void *server(void *arg);
  static void *reader(void *arg);
  static void *compute(void *arg);
  static void *pull(void *arg);
  static void *push(void *arg);
  uint32_t word_index(const std::string &word);
  bool next_example(
    std::vector<uint32_t> &input,
    uint32_t &target,
    std::string &cur_shard_path,
    std::ifstream &cur_shard,
    std::vector<std::string> cur_line,
    int &cur_index);

 protected:
  std::unique_ptr<distrust::ParamServiceClient> param_client_;

  // The language model holds all our parameters
  distrust::ModelInfo model_info_;
  std::unordered_map<std::string, uint32_t> vocab_;
  std::unique_ptr<LanguageModel> model_;
  pthread_mutex_t model_lock_;

  // Threads
  pthread_t server_thread_;    // worker service
  pthread_t announce_thread_;  // announce to master
  pthread_t pull_thread_;      // parameter pulling
  pthread_t push_thread_;      // update pushing
  pthread_t reader_thread_;    // data reading
  pthread_t compute_thread_;   // update computation

  // Connection info
  std::string master_ip_;
  int master_port_, worker_port_;

  // Paths to training data
  std::queue<std::string> shard_paths_;
  pthread_mutex_t shard_paths_lock_;

  // Paths to all shards completed by worker
  std::unordered_set<std::string> completed_shards_;
  pthread_mutex_t completed_shards_lock_;

  // Learning rate
  double learn_rate_;

  // Minibatch size
  int batch_size_;

  // Server status
  bool server_ready_;
  pthread_mutex_t server_ready_lock_;
  pthread_cond_t server_ready_cond_;

  // Control flag(s)
  bool stop_;
  pthread_mutex_t stop_lock_;
  pthread_cond_t stop_cond_;
};

class WorkerServiceHandler : virtual public distrust::WorkerServiceIf {

 public:
  WorkerServiceHandler(Worker *worker) : worker_(worker) { }
  ~WorkerServiceHandler() { }

  void heartbeat(distrust::HBResponse &_return);
  void start(const distrust::StartRequest &request);
  void stop();
  void reassign(const std::vector<std::string> &shard_paths);

 protected:
  Worker *worker_;
};

#endif
