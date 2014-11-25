#include <iostream>
#include <getopt.h>

#include "Worker.h"
#include <transport/TSocket.h>
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
  master_ip_(master_ip),
  master_port_(master_port) {

  shared_ptr<TSocket> socket(new TSocket(master_ip.data(), master_port));
  shared_ptr<TTransport> transport(new TBufferedTransport(socket));
  shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
  param_client_ = std::unique_ptr<ParamServiceClient>(
    new ParamServiceClient(protocol));

  while (true) {
    try {
      transport->open();
      break;
    } catch (const TException &tx) {
      std::cerr << "Transport error: " << tx.what() << std::endl;
    }
    sleep(1);
  }

  std::cout << "Connected to param server at " << master_ip << ":" << master_port << std::endl;
}

WorkerServiceHandler::~WorkerServiceHandler() { }

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
    {"help", no_argument, 0, 'h'},
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
      case 'h':
        usage();
        exit(0);
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

  shared_ptr<WorkerServiceHandler> handler(new WorkerServiceHandler(master_ip, master_port));
  shared_ptr<TProcessor> processor(new WorkerServiceProcessor(handler));
  shared_ptr<TServerTransport> serverTransport(new TServerSocket(worker_port));
  shared_ptr<TTransportFactory> transportFactory(new TBufferedTransportFactory());
  shared_ptr<TProtocolFactory> protocolFactory(new TBinaryProtocolFactory());

  TSimpleServer server(processor, serverTransport, transportFactory, protocolFactory);
  server.serve();
  return 0;
}
