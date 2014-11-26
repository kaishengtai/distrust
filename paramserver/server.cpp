#include "Server.h"

#include <iostream>
#include <getopt.h>
#include <sstream>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TThreadedServer.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TBufferTransports.h>

using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;
using namespace apache::thrift::server;

using LogCabin::Client::Cluster;
using LogCabin::Client::Tree;

using boost::shared_ptr;

using namespace distrust;

std::string
get_worker_key(const std::string &ip, const int32_t port) {
  std::stringstream ss;
  ss << ip << port;
  return ss.str();
}

ParamServer::ParamServer(const int32_t port, const std::string &raft_cluster) :
    port_(port),
    cluster_(raft_cluster) { }

void 
ParamServer::run() {
  int server_ret = pthread_create(&server_thread_, NULL, &ParamServer::server, this);
  pthread_join(server_thread_, NULL);
}

void
ParamServer::add_worker(const std::string &ip, const int32_t port) {
  shared_ptr<TSocket> socket(new TSocket(ip, port));
  shared_ptr<TTransport> transport(new TBufferedTransport(socket));
  shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
  worker_clients_[get_worker_key(ip, port)] =
    std::unique_ptr<WorkerServiceClient>(new WorkerServiceClient(protocol));

  // Make the connection
  while (true) {
    try {
      transport->open();
      break;
    } catch (const TException &tx) {
      std::cerr << "Transport error: " << tx.what() << std::endl;
    }
    sleep(1);
  }
  std::cout << "Connected to " << ip << ":" << port << std::endl;
}

void *
ParamServer::server(void * arg) {
  ParamServer *context = (ParamServer *)arg;

  shared_ptr<ParamServiceHandlerFactory> handlerFactory(
    new ParamServiceHandlerFactory(context));
  shared_ptr<TProcessorFactory> processorFactory(
    new ParamServiceProcessorFactory(handlerFactory));
  shared_ptr<TServerTransport> serverTransport(
    new TServerSocket(context->port_));
  shared_ptr<TTransportFactory> transportFactory(
    new TBufferedTransportFactory());
  shared_ptr<TProtocolFactory> protocolFactory(new TBinaryProtocolFactory());

  TThreadedServer server(
    processorFactory, serverTransport, transportFactory, protocolFactory);
  server.serve();
  return NULL;
}

void
ParamServiceHandler::announce(
  AnnounceResponse& _return,
  const int32_t worker_port) {

  // Open reverse connection
  std::cout << "Announce from " << worker_ip_ << std::endl;
  server_->add_worker(worker_ip_, worker_port);

  // Return parameters
  // TODO: replace hardcoded values
  _return.model_info.window_size = 5;
  _return.model_info.vocab_size = 40000;
  _return.model_info.start_token_index = 1;
  _return.model_info.end_token_index = 2;
  _return.model_info.wordvec_dim = 128;
  _return.model_info.hidden_dim = 256;

  _return.params.wordvec_weights.push_back(1.0);
  _return.params.input_hidden_weights.push_back(1.0);
  _return.params.input_hidden_biases.push_back(1.0);
  _return.params.hidden_output_weights.push_back(1.0);
  _return.params.hidden_output_biases.push_back(1.0);

  _return.shard_paths.push_back("/foo/bar/");
  _return.shard_paths.push_back("/foo/baz/");

  _return.learn_rate = 0.1;
}

void
ParamServiceHandler::push_update(const Params& params) {
  printf("push_update\n");
}

void
ParamServiceHandler::pull_params(Params& _return) {
  printf("pull_params\n");
}

/**
 * Parses argv for the main function.
 */
class OptionParser {
 public:
  OptionParser(int& argc, char**& argv)
      : argc_(argc),
        argv_(argv),
        cluster("logcabin:61023"),
        port(0) {
    static struct option longOptions[] = {
      {"cluster",  required_argument, NULL, 'c'},
      {"port", required_argument, 0, 'p'},
      {"data", required_argument, 0, 'd'},
      {"help",  no_argument, NULL, 'h'},
      {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "c:p:d:h", longOptions, NULL)) != -1) {
      switch (c) {
        case 'c':
          cluster = optarg;
          break;
        case 'p':
          port = atoi(optarg);
          break;
        case 'd':
          data_dir = optarg;
          break;
        case 'h':
          usage();
          exit(0);
        case '?':
        default:
          usage();
          exit(1);
      }
    }
  }

  void usage() {
      std::cout << "Usage: " << argv_[0] << " [options]"
                << std::endl;
      std::cout << "Options: " << std::endl;
      std::cout << "  -c, --cluster <address> "
                << "The network address of the LogCabin cluster "
                << "(default: logcabin:61023)" << std::endl;
      std::cout << "  -p, --port <port> "
                << "The port that the ParamServer listens to RPC" << std::endl;
      std::cout << "  -d, --data <dir> "
                << "The path to the directory that contains the dataset "
                << "(assumed to contain vocabulary in vocab.txt)" << std::endl;
      std::cout << "  -h, --help              "
                << "Print this usage information" << std::endl;
  }

  int& argc_;
  char**& argv_;
  std::string cluster;
  int port;
  std::string data_dir;
};


int main(int argc, char **argv) {
  // The following trivial code tests whether LogCabin is up and running.
  Cluster cluster("logcabin:61023");
  Tree tree = cluster.getTree();
  tree.makeDirectoryEx("/etc");
  tree.writeEx("/etc/passwd", "ha");
  std::string contents = tree.readEx("/etc/passwd");
  printf("Contents of logcabin at %s: %s\n", "/etc/passwd", contents.c_str());
  tree.removeDirectoryEx("/etc");

  OptionParser options(argc, argv);
  
  printf("paramserver: using LogCabin cluster: %s\n", options.cluster.c_str());
  printf("paramserver: start serving on port %d\n", options.port);
  ParamServer param_server(options.port, options.cluster);
  param_server.run();
  return 0;
}

