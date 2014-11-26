#ifndef SERVER_H
#define SERVER_H

#include <pthread.h>
#include <unordered_map>
#include <transport/TSocket.h>

#include "distrust/gen-cpp/ParamService.h"
#include "distrust/gen-cpp/WorkerService.h"
#include "logcabin/Client/Client.h"

using apache::thrift::TConnectionInfo;
using apache::thrift::transport::TSocket;
using distrust::WorkerServiceClient;

class ParamServer {
 friend class ParamServiceHandler;

 public:
  ParamServer(const int32_t port, const std::string &raft_cluster);
  ~ParamServer() { }
  void run();

 protected:
  void add_worker(const std::string &ip, const int32_t port);
  static void *server(void *);

 protected:
  int port_;
  pthread_t server_thread_;
  LogCabin::Client::Cluster cluster_;
  std::unordered_map<std::string, std::unique_ptr<WorkerServiceClient>>
    worker_clients_;
};

class ParamServiceHandler : virtual public distrust::ParamServiceIf {
 public:
  ParamServiceHandler(ParamServer *server, const std::string &worker_ip)
    : server_(server),
      worker_ip_(worker_ip) { }

  void announce(distrust::AnnounceResponse &_return, const int32_t worker_port);
  void push_update(const distrust::Params &params);
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
      boost::dynamic_pointer_cast<TSocket>(
        connInfo.transport);
    return new ParamServiceHandler(server_, socket->getPeerAddress());
  }

  void releaseHandler(distrust::ParamServiceIf* handler) {
    delete handler;
  }
 
 protected:
  ParamServer *server_;
};

#endif
