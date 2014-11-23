// This autogenerated skeleton file illustrates how to build a server.
// You should copy it to another filename to avoid overwriting it.

#include "ParamService.h"
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

class ParamServiceHandler : virtual public ParamServiceIf {
 public:
  ParamServiceHandler() {
    // Your initialization goes here
  }

  void announce(AnnounceResponse& _return, const AnnounceRequest& request) {
    // Your implementation goes here
    printf("announce\n");
  }

  void push_update(UpdateResponse& _return, const UpdateRequest& request) {
    // Your implementation goes here
    printf("push_update\n");
  }

  void pull_params(PullResponse& _return, const PullRequest& request) {
    // Your implementation goes here
    printf("pull_params\n");
  }

};

int main(int argc, char **argv) {
  int port = 9090;
  shared_ptr<ParamServiceHandler> handler(new ParamServiceHandler());
  shared_ptr<TProcessor> processor(new ParamServiceProcessor(handler));
  shared_ptr<TServerTransport> serverTransport(new TServerSocket(port));
  shared_ptr<TTransportFactory> transportFactory(new TBufferedTransportFactory());
  shared_ptr<TProtocolFactory> protocolFactory(new TBinaryProtocolFactory());

  TSimpleServer server(processor, serverTransport, transportFactory, protocolFactory);
  server.serve();
  return 0;
}

