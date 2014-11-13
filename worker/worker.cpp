#include <iostream>

#include "../gen-cpp/Worker.h"
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TSimpleServer.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TBufferTransports.h>

using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;
using namespace apache::thrift::server;

using boost::shared_ptr;

using namespace distrust;

class WorkerHandler : virtual public WorkerIf {
 public:
  WorkerHandler() {
    // Your initialization goes here
  }

  void heartbeat(HBResponse& _return, const HBRequest& request) {
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

};

int main(int argc, char **argv) {
  int port = 9090;
  shared_ptr<WorkerHandler> handler(new WorkerHandler());
  shared_ptr<TProcessor> processor(new WorkerProcessor(handler));
  shared_ptr<TServerTransport> serverTransport(new TServerSocket(port));
  shared_ptr<TTransportFactory> transportFactory(new TBufferedTransportFactory());
  shared_ptr<TProtocolFactory> protocolFactory(new TBinaryProtocolFactory());

  TSimpleServer server(processor, serverTransport, transportFactory, protocolFactory);
  server.serve();
  return 0;
}