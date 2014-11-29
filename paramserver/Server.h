#ifndef SERVER_H
#define SERVER_H

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
    const std::string &data_dir);
  ~ParamServer() { }
  void run();

 protected:
  void add_worker(const std::string &ip, const int32_t port);
  void read_vocab(const std::string &path);
  static void *server(void *);
  static void *heartbeat(void *);
  static void *backup_params(void *);

 protected:
  // ParamServer port
  int32_t port_;

  // Raft cluster
  LogCabin::Client::Cluster cluster_;

  // Main thread handle
  pthread_t server_thread_;
  
  // Backup parameters thread handle.  
  pthread_t backup_thread_;
  
  // Heartbeat thread handles (one per worker).
  std::unordered_map<std::string, pthread_t> heartbeat_threads_;
  
  // Worker RPC stubs.
  std::unordered_map<std::string, std::unique_ptr<WorkerServiceClient>>
    worker_clients_;

  // Paths to data shards
  std::vector<std::string> shard_paths_;

  // Language model
  distrust::ModelInfo model_info_;
  std::unique_ptr<LanguageModel> model_;
};

}  // namespace distrust

#endif  // SERVER_H
