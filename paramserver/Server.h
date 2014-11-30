#ifndef SERVER_H
#define SERVER_H

#include <mutex>
#include <pthread.h>
#include <unordered_map>

#include "../LanguageModel.h"
#include "distrust/gen-cpp/WorkerService.h"
#include "logcabin/Client/Client.h"

namespace distrust {

class ParamServer {
 friend class ParamServiceHandler;

 public:
  ParamServer(
    const int32_t window_size,
    const int32_t wordvec_dim,
    const int32_t hidden_dim,
    const int32_t port,
    const std::string &raft_cluster,
    const std::string &train_dir,
    const std::string &test_dir);
  ~ParamServer() { }
  void run();

 protected:
  void reshard();
  void add_worker(const std::string &ip, const int32_t port);
  void read_vocab(const std::string &path);
  std::vector<std::string> get_shard_paths(
    const std::string &dir, const std::string &vocab_path);
  static uint32_t time_millis();
  static void *server(void *);
  static void *heartbeat(void *);
  static void *backup_params(void *);
  static void *test_model(void *);

 protected:
  // ParamServer port
  int32_t port_;

  // Raft cluster
  LogCabin::Client::Cluster cluster_;

  // Main thread handle
  pthread_t server_thread_;
  
  // Backup parameters thread handle.  
  pthread_t backup_thread_;

  // Model testing thread handle
  pthread_t test_thread_;
  
  // Heartbeat thread handles (one per worker).
  std::mutex heartbeat_threads_lock_;
  std::unordered_map<std::string, pthread_t> heartbeat_threads_;
  
  // Worker RPC stubs.
  std::mutex worker_clients_lock_;
  std::unordered_map<std::string, std::unique_ptr<WorkerServiceClient>>
    worker_clients_;

  // Paths to train data shards
  std::mutex shard_paths_lock_;
  std::vector<std::string> shard_paths_;
  std::unordered_map<std::string, std::vector<std::string>> worker_to_shards_;

  // Paths to test data shards
  std::vector<std::string> test_shard_paths_;

  // Language model
  std::mutex model_lock_;
  distrust::ModelInfo model_info_;
  std::unique_ptr<LanguageModel> model_;
};

}  // namespace distrust

#endif  // SERVER_H
