#include <iostream>
#include <getopt.h>

#include "Worker.h"
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

WorkerServiceHandler::WorkerServiceHandler(
    const std::string& master_ip,
    int master_port) :
  _master_ip(master_ip),
  _master_port(master_port)  {

  

}

WorkerServiceHandler::~WorkerServiceHandler() {

}

void
WorkerServiceHandler::heartbeat(
  HBResponse& _return,
  const HBRequest& request) {

}

void
WorkerServiceHandler::start(const StartRequest& request) {

}

void
WorkerServiceHandler::stop() {

}

void
WorkerServiceHandler::reassign(const ReassignRequest& request) {

}

void
usage() {
  std::cerr << "./Worker -i/--masterip <ip> -p/--masterport <port> -w/--workerport <port>" << std::endl;
}

int
main(int argc, char **argv) {
  std::string master_ip = "";
  int master_port = -1;
  int worker_port = WORKER_DEFAULT_PORT;

  static struct option long_options[] = {
    {"masterip",   required_argument, 0, 'i'},
    {"masterport", required_argument, 0, 'p'},
    {"workerport", required_argument, 0, 'w'},
    {0, 0, 0, 0}
  };
  int option_index = 0;

  int c;
  while ((c = getopt_long(argc, argv, "i:p:w:", long_options, &option_index)) != -1) {
    switch (c) {
      case 'i':
        master_ip = optarg;
        break;
      case 'p':
        master_port = atoi(optarg);
        break;
      case 'w':
        worker_port = atoi(optarg);
        break;
      default:
        usage();
        exit(1);
    }
  }

  if (master_ip.empty() || master_port == -1) {
    usage();
    exit(1);
  }

  std::cout << "Starting worker on port " << worker_port << std::endl;
  std::cout << "Master IP: " << master_ip << std::endl;
  std::cout << "Master port: " << master_port << std::endl;

  shared_ptr<WorkerServiceHandler> handler(new WorkerServiceHandler(master_ip, master_port));
  shared_ptr<TProcessor> processor(new WorkerServiceProcessor(handler));
  shared_ptr<TServerTransport> serverTransport(new TServerSocket(worker_port));
  shared_ptr<TTransportFactory> transportFactory(new TBufferedTransportFactory());
  shared_ptr<TProtocolFactory> protocolFactory(new TBinaryProtocolFactory());

  TSimpleServer server(processor, serverTransport, transportFactory, protocolFactory);
  server.serve();
  return 0;
}
