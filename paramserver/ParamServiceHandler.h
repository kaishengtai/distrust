#ifndef PARAMSERVICEHANDLER_H
#define PARAMSERVICEHANDLER_H

#include "Server.h"

#include <transport/TSocket.h>
#include "distrust/gen-cpp/ParamService.h"

using apache::thrift::TConnectionInfo;
using apache::thrift::transport::TSocket;

namespace distrust {

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

class ParamServiceHandlerFactory :
    virtual public distrust::ParamServiceIfFactory {
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

}  // namespace distrust

#endif  // PARAMSERVICEHANDLER_H
