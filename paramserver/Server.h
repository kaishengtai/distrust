#ifndef SERVER_H
#define SERVER_H

#include <pthread.h>
#include <unordered_map>
#include <transport/TSocket.h>

#include "../LanguageModel.h"
#include "distrust/gen-cpp/ParamService.h"
#include "distrust/gen-cpp/WorkerService.h"
#include "logcabin/Client/Client.h"

using apache::thrift::TConnectionInfo;
using apache::thrift::transport::TSocket;
using distrust::WorkerServiceClient;

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

 protected:
  // ParamServer port
  int32_t port_;

  // Raft cluster
  LogCabin::Client::Cluster cluster_;

  pthread_t server_thread_;
  std::unordered_map<std::string, pthread_t> heartbeat_threads_;
  std::unordered_map<std::string, std::unique_ptr<WorkerServiceClient>>
    worker_clients_;

  // Paths to data shards
  std::vector<std::string> shard_paths_;

  // Language model
  distrust::ModelInfo model_info_;
  std::unique_ptr<LanguageModel> model_;
};

class ParamServiceHandler : virtual public distrust::ParamServiceIf {
 public:
  ParamServiceHandler(ParamServer *server, const std::string &worker_ip)
    : server_(server),
      worker_ip_(worker_ip) { }

  void announce(distrust::AnnounceResponse &_return, const int32_t worker_port);
  void push_update(const distrust::ParamUpdate &update);
  void pull_params(distrust::Params &_return);

 protected:
  ParamServer *server_;
  std::string worker_ip_;
};

class ParamServiceHandlerFactory : virtual public distrust::ParamServiceIfFactory {
 public:
  ParamServiceHandlerFactory(ParamServer *server) : server_(server) { }

  // This can be used to open a reverse connection from paramserver to worker
  ParamServiceHandler* getHandler(const TConnectionInfo& connInfo) {
    boost::shared_ptr<TSocket> socket = 
      boost::dynamic_pointer_cast<TSocket>(connInfo.transport);
    return new ParamServiceHandler(server_, socket->getPeerAddress());
  }

  void releaseHandler(distrust::ParamServiceIf* handler) {
    delete handler;
  }
 
 protected:
  ParamServer *server_;
};

#endif
