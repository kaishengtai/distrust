// This autogenerated skeleton file illustrates how to build a server.
// You should copy it to another filename to avoid overwriting it.

#include "WorkerService.h"
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TSimpleServer.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TBufferTransports.h>

using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;
using namespace ::apache::thrift::server;

using boost::shared_ptr;

using namespace  ::distrust;

class WorkerServiceHandler : virtual public WorkerServiceIf {
 public:
  WorkerServiceHandler() {
    // Your initialization goes here
  }

  void heartbeat(HBResponse& _return) {
    // Your implementation goes here
    printf("heartbeat\n");
  }

  void start(const StartRequest& request) {
    // Your implementation goes here
    printf("start\n");
  }

  void stop() {
    // Your implementation goes here
    printf("stop\n");
  }

  void reassign(const std::vector<std::string> & shard_paths) {
    // Your implementation goes here
    printf("reassign\n");
  }

};

int main(int argc, char **argv) {
  int port = 9090;
  shared_ptr<WorkerServiceHandler> handler(new WorkerServiceHandler());
  shared_ptr<TProcessor> processor(new WorkerServiceProcessor(handler));
  shared_ptr<TServerTransport> serverTransport(new TServerSocket(port));
  shared_ptr<TTransportFactory> transportFactory(new TBufferedTransportFactory());
  shared_ptr<TProtocolFactory> protocolFactory(new TBinaryProtocolFactory());

  TSimpleServer server(processor, serverTransport, transportFactory, protocolFactory);
  server.serve();
  return 0;
}

