#include "Server.h"

#include <iostream>
#include <fstream>
#include <getopt.h>
#include <sstream>

#include <boost/filesystem.hpp>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TThreadedServer.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TBufferTransports.h>

using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;
using namespace apache::thrift::server;
namespace fs = boost::filesystem;

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

ParamServer::ParamServer(
  const int32_t port,
  const std::string &raft_cluster,
  const std::string &data_dir) :
    port_(port),
    cluster_(raft_cluster) { 

  if (!fs::exists(data_dir) || !fs::is_directory(data_dir)) {
    throw std::invalid_argument("Invalid data directory: " + data_dir);
  }

  // read vocabulary
  fs::path vocab_file("vocab.txt");
  std::string vocab_path = (data_dir / vocab_file).native();
  if (!fs::exists(vocab_path)) {
    throw std::invalid_argument("No vocab.txt in data directory: " + data_dir);
  }
  read_vocab(vocab_path);

  // get shard paths from data directory
  fs::directory_iterator end_iter;
  for (fs::directory_iterator dir_iter(data_dir); dir_iter != end_iter; dir_iter++) {
    if (fs::is_regular_file(dir_iter->status())) {
      std::string path = dir_iter->path().native();
      if (path == vocab_path) continue;
      shard_paths_.push_back(path);
    }
  }
}

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

void
ParamServer::read_vocab(const std::string &path) {
  std::cout << "Reading vocabulary from " << path << std::endl;
  std::ifstream file(path);
  std::string word;
  std::string START("<s>");
  std::string END("</s>");
  std::string UNK("<unk>");

  int i = 0;
  int start_idx = -1;
  int end_idx = -1;
  int unk_idx = -1;
  while (file >> word) {
    if (word == START) {
      start_idx = i;
    } else if (word == END) {
      end_idx = i;
    } else if (word == UNK) {
      unk_idx = i;
    }

    vocab_.push_back(word);
    i++;
  }

  if (start_idx == -1) {
    vocab_.push_back(START);
    start_idx = i++;
  }

  if (end_idx == -1) {
    vocab_.push_back(END);
    end_idx = i++;
  }

  if (unk_idx == -1) {
    vocab_.push_back(UNK);
    unk_idx = i++;
  }

  start_token_index_ = start_idx;
  end_token_index_ = end_idx;
  unk_token_index_ = unk_idx;
  file.close();
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
  _return.model_info.wordvec_dim = 128;
  _return.model_info.hidden_dim = 256;
  _return.model_info.start_token_index = server_->start_token_index_;
  _return.model_info.end_token_index = server_->end_token_index_;
  _return.model_info.unk_token_index = server_->unk_token_index_;
  _return.model_info.vocab = server_->vocab_;

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
  ParamServer param_server(options.port, options.cluster, options.data_dir);
  param_server.run();
  return 0;
}

