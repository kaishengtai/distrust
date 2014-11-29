#include "Server.h"
#include "ParamServiceHandler.h"

#include "logcabin/Client/Client.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/filesystem.hpp>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TThreadedServer.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TBufferTransports.h>
#include <transport/TSocket.h>

using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;
using namespace apache::thrift::server;
namespace fs = boost::filesystem;

using LogCabin::Client::Cluster;
using LogCabin::Client::Tree;

using boost::shared_ptr;

namespace distrust {

namespace {

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

}  // namespace

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
  std::unique_lock<std::mutex> shard_paths_write_lock(shard_paths_lock_);
  fs::directory_iterator end_iter;
  for (fs::directory_iterator dir_iter(data_dir);
       dir_iter != end_iter;
       dir_iter++) {
    if (fs::is_regular_file(dir_iter->status())) {
      std::string path = dir_iter->path().native();
      if (path == vocab_path) continue;
      shard_paths_.push_back(path);
    }
  }
  shard_paths_write_lock.unlock();

  // initialize parameters
  std::unique_lock<std::mutex> model_write_lock(model_lock_);
  model_info_.window_size = window_size;
  model_info_.wordvec_dim = wordvec_dim;
  model_info_.hidden_dim = hidden_dim;
  model_ = std::unique_ptr<LanguageModel>(new LanguageModel(model_info_));
  model_->random_init();
  model_write_lock.unlock();

  // set up logcabin directories
  Tree tree = cluster_.getTree();
  tree.makeDirectoryEx("/data");

  // launch backup params thread
  pthread_create(&backup_thread_, NULL, &ParamServer::backup_params, this);
}

// Protected functions

struct HeartbeatConfig {
  ParamServer *server;
  std::string ip;
  int32_t port;
};

void
ParamServer::reshard() {
  // Reassign worker shards -- for now, naively redivide the shards uniformly.
  // TODO: we need to keep track of completed shards (and mini-shards?).
  
  // We need a read lock on worker_clients_ and a write lock on shard_paths_,
  // without deadlock.
  std::unique_lock<std::mutex> worker_clients_read_lock(
      worker_clients_lock_, std::defer_lock);
  std::unique_lock<std::mutex> shard_paths_write_lock(
      shard_paths_lock_, std::defer_lock);
  // This prevents deadlock.
  std::lock(worker_clients_read_lock, shard_paths_write_lock);
  int num_workers = worker_clients_.size();
  if (num_workers <= 0) {
    return;
  }
  int shards_per_worker = std::max(
      static_cast<int>(shard_paths_.size() / num_workers), 1);
  int shard_idx = 0;
  int worker_idx = 0;
  for (const auto &kv : worker_clients_) {
    worker_to_shards_[kv.first].clear();
    int shard_upper_bound = (worker_idx == (num_workers - 1)) ?
        shard_paths_.size() : shard_idx + shards_per_worker;
    std::cout << "Reassigning worker [" << kv.first << "] to shards "
              << shard_idx << " to " << shard_upper_bound - 1 << std::endl;
    for (int shard = shard_idx; shard < shard_upper_bound; ++shard) {
      worker_to_shards_[kv.first].push_back(shard_paths_[shard]);
    }
    
    // TODO: error handling for RPC call.
    kv.second->reassign(worker_to_shards_[kv.first]);
    shard_idx = shard_upper_bound;
    ++worker_idx;
  }
  // Unlocked in destructors.
}

void
ParamServer::add_worker(const std::string &ip, const int32_t port) {
  shared_ptr<TSocket> socket(new TSocket(ip, port));
  shared_ptr<TTransport> transport(new TBufferedTransport(socket));
  shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
  std::string worker_key = get_worker_key(ip, port);
  std::unique_lock<std::mutex> worker_clients_write_lock(worker_clients_lock_);
  worker_clients_[worker_key] =
    std::unique_ptr<WorkerServiceClient>(new WorkerServiceClient(protocol));
  worker_clients_write_lock.unlock();

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

  std::unique_lock<std::mutex> heartbeat_threads_write_lock(
      heartbeat_threads_lock_);
  pthread_create(&heartbeat_threads_[worker_key], NULL,
                 &ParamServer::heartbeat, heartbeat_config);
  heartbeat_threads_write_lock.unlock();

  // Reshard and reassign
  reshard();
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

// Public functions

void 
ParamServer::run() {
  // Start server RPC serving thread.
  int server_ret = pthread_create(&server_thread_, NULL, &ParamServer::server,
                                  this);
                                  
  // Main process thread waits indefinitely.
  pthread_join(server_thread_, NULL);
}

// Static functions

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
  std::unique_lock<std::mutex> worker_clients_read_lock(
      heartbeat_config->server->worker_clients_lock_);
  WorkerServiceClient *worker_client =
      heartbeat_config->server->worker_clients_[worker_key].get();
  worker_clients_read_lock.unlock();

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
  std::unique_lock<std::mutex> worker_clients_write_lock(
      heartbeat_config->server->worker_clients_lock_);
  // Remove worker client stub.
  heartbeat_config->server->worker_clients_.erase(worker_key);
  worker_clients_write_lock.unlock();
  // Reshard.
  heartbeat_config->server->reshard();
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

}  // namespace distrust

