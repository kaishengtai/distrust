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

std::string serialize_params(const Params &obj) {
  shared_ptr<TMemoryBuffer> transportOut(new TMemoryBuffer());
  shared_ptr<TBinaryProtocol> protocolOut(new TBinaryProtocol(transportOut));
  obj.write(protocolOut.get());
  std::string serialized_string = transportOut->getBufferAsString();
  return serialized_string;
}

void deserialize_params(const std::string &serialized_string, Params *params) {
  char *buf = new char[serialized_string.size()];
  strncpy(buf, serialized_string.c_str(), serialized_string.size());
  shared_ptr<TMemoryBuffer> transportIn(
      new TMemoryBuffer((uint8_t *)buf, (uint32_t)serialized_string.size()));
  shared_ptr<TBinaryProtocol> protocolIn(new TBinaryProtocol(transportIn));
  params->read(protocolIn.get());
  delete[] buf;
}

struct HeartbeatConfig {
  ParamServer *server;
  std::string ip;
  int32_t port;
};

ParamServer::ParamServer(
  const int32_t window_size,
  const int32_t wordvec_dim,
  const int32_t hidden_dim,
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

  // initialize parameters
  model_info_.window_size = window_size;
  model_info_.wordvec_dim = wordvec_dim;
  model_info_.hidden_dim = hidden_dim;
  model_ = std::unique_ptr<LanguageModel>(new LanguageModel(model_info_));
  model_->random_init();
  
  // set up logcabin directories
  Tree tree = cluster_.getTree();
  tree.makeDirectoryEx("/data");
  
  // launch backup params thread
  pthread_create(&backup_thread_, NULL, &ParamServer::backup_params, this);
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
  std::string worker_key = get_worker_key(ip, port);
  worker_clients_[worker_key] =
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
  
  // Spin up a watcher thread to periodically monitor heartbeats.
  HeartbeatConfig *heartbeat_config = new HeartbeatConfig;
  heartbeat_config->server = this;
  heartbeat_config->ip = ip;
  heartbeat_config->port = port;

  pthread_create(&heartbeat_threads_[worker_key], NULL,
                 &ParamServer::heartbeat, heartbeat_config);
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

    model_info_.vocab.push_back(word);
    i++;
  }

  if (start_idx == -1) {
    model_info_.vocab.push_back(START);
    start_idx = i++;
  }

  if (end_idx == -1) {
    model_info_.vocab.push_back(END);
    end_idx = i++;
  }

  if (unk_idx == -1) {
    model_info_.vocab.push_back(UNK);
    unk_idx = i++;
  }

  model_info_.start_token_index = start_idx;
  model_info_.end_token_index = end_idx;
  model_info_.unk_token_index = unk_idx;
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

void *
ParamServer::heartbeat(void * arg) {
  std::unique_ptr<HeartbeatConfig> heartbeat_config((HeartbeatConfig *)arg);
  std::string worker_key = get_worker_key(heartbeat_config->ip,
                                          heartbeat_config->port);
  WorkerServiceClient *worker_client =
      heartbeat_config->server->worker_clients_[worker_key].get();

  std::cout << "Starting heartbeat monitoring thread for worker "
            << heartbeat_config->ip << ":" << heartbeat_config->port
            << std::endl;

  HBResponse hbresponse;
  while (true) {
    // Ping
    try {
      worker_client->heartbeat(hbresponse);
    } catch (TException &tx) {
      std::cout << "ERROR: " << tx.what() << std::endl;
      std::cout << "Worker " << heartbeat_config->ip << ":"
                << heartbeat_config->port << " did not return heartbeat."
                << std::endl;
      // Break out to try
      break;
    }

    sleep(5);
  }
  
  // Handle stopping worker, reassigning work to other workers, etc.
  std::cout << "Handling reassignment" << std::endl;
  return NULL;
}

void *
ParamServer::backup_params(void * arg) {
  ParamServer *context = (ParamServer *)arg;
  
  while (true) {
    std::cout << "Backing up params to LogCabin." << std::endl;

    // Write parameters to cluster
    Params params;
    context->model_->get_params(params);
    std::string serialized = serialize_params(params);
    Tree tree = context->cluster_.getTree();
    
    // Currently, the serialized params exceeds the max message size. Right now,
    // to demo that backup works, I'm truncating the serialized string to the
    // first 1024 bytes.
    tree.writeEx("/data/params", serialized.substr(0, 1024));
    std::string contents = tree.readEx("/data/params");
    assert(contents == serialized.substr(0, 1024));
    
    sleep(20);
  }
}

void
ParamServiceHandler::announce(
  AnnounceResponse& _return,
  const int32_t worker_port) {

  // Open reverse connection
  std::cout << "Announce from " << worker_ip_ << std::endl;
  server_->add_worker(worker_ip_, worker_port);

  // Return parameters
  _return.model_info = server_->model_info_;
  _return.shard_paths = server_->shard_paths_;
  _return.learn_rate = 0.1;
  _return.batch_size = 128;
  server_->model_->get_params(_return.params);
}

void
ParamServiceHandler::push_update(const ParamUpdate &update) {
  printf("push_update\n");
}

void
ParamServiceHandler::pull_params(Params &_return) {
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
        window_size(3),
        wordvec_dim(50),
        hidden_dim(100),
        port(8000),
        cluster("logcabin:61023") {
    static struct option longOptions[] = {
      {"cluster",  required_argument, NULL, 'c'},
      {"port", required_argument, 0, 'p'},
      {"data", required_argument, 0, 'd'},
      {"window", required_argument, 0, 'w'},
      {"wordvec", required_argument, 0, 'v'},
      {"hidden", required_argument, 0, 'H'},
      {"help",  no_argument, NULL, 'h'},
      {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "c:p:d:w:v:H:h", longOptions, NULL)) != -1) {
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
        case 'w':
          window_size = atoi(optarg);
          break;
        case 'v':
          wordvec_dim = atoi(optarg);
          break;
        case 'H':
          hidden_dim = atoi(optarg);
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
      std::cout << "  -w, --window <dim>" << std::endl;
      std::cout << "  -v, --wordvec <dim>" << std::endl;
      std::cout << "  -H, --hidden <dim>" << std::endl;
      std::cout << "  -h, --help              "
                << "Print this usage information" << std::endl;
  }

  int& argc_;
  char**& argv_;

  int window_size;
  int wordvec_dim;
  int hidden_dim;
  int port;
  std::string cluster;
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
  ParamServer param_server(
    options.window_size,
    options.wordvec_dim,
    options.hidden_dim,
    options.port,
    options.cluster,
    options.data_dir);
  param_server.run();
  return 0;
}

