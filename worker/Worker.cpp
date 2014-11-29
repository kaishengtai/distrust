#include "Worker.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <getopt.h>

#include <boost/regex.hpp>

#include <transport/TSocket.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TThreadedServer.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TBufferTransports.h>

using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;
using namespace apache::thrift::server;

using boost::shared_ptr;

using namespace distrust;

Worker::Worker(
  const std::string& master_ip,
  const int master_port,
  const int worker_port) :

  master_ip_(master_ip),
  master_port_(master_port),
  worker_port_(worker_port),
  server_ready_(false),
  stop_(true) {

  pthread_mutex_init(&model_lock_, NULL);
  pthread_mutex_init(&stop_lock_, NULL);
  pthread_mutex_init(&shard_paths_lock_, NULL);
  pthread_mutex_init(&completed_shards_lock_, NULL);
  pthread_mutex_init(&server_ready_lock_, NULL);
  pthread_cond_init(&server_ready_cond_, NULL);
  pthread_cond_init(&stop_cond_, NULL);
  pthread_cond_init(&has_shards_cond_, NULL);

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

void *
Worker::server(void *arg) {
  Worker *self = (Worker *) arg;
  pthread_mutex_lock(&self->server_ready_lock_);
  const int worker_port = self->worker_port_;
  std::cout << "Starting worker on port " << worker_port << std::endl;
  shared_ptr<WorkerServiceHandler> handler(new WorkerServiceHandler(self));
  shared_ptr<TProcessor> processor(new WorkerServiceProcessor(handler));
  shared_ptr<TServerTransport> serverTransport(new TServerSocket(worker_port));
  shared_ptr<TTransportFactory> transportFactory(new TBufferedTransportFactory());
  shared_ptr<TProtocolFactory> protocolFactory(new TBinaryProtocolFactory());
  TThreadedServer server(processor, serverTransport, transportFactory, protocolFactory);
  self->server_ready_ = true;
  pthread_cond_signal(&self->server_ready_cond_);
  pthread_mutex_unlock(&self->server_ready_lock_);
  server.serve();
  return NULL;
}

void *
Worker::announce(void *arg) {
  Worker *self = (Worker *) arg;

  // Wait until the server signals that it's starting up.
  pthread_mutex_lock(&self->server_ready_lock_);
  while (!self->server_ready_) {
    pthread_cond_wait(&self->server_ready_cond_, &self->server_ready_lock_);
  }
  pthread_mutex_unlock(&self->server_ready_lock_);

  // Give the server time to start up
  sleep(1);

  // Announce the worker to the master.
  // No requests from the master will be received until after this point.
  pthread_mutex_lock(&self->stop_lock_);
  AnnounceResponse resp;
  self->param_client_->announce(resp, self->worker_port_);

  // Initialize model parameters.
  // Takes ownership of memory holding parameters.
  pthread_mutex_lock(&self->model_lock_);
  self->model_info_ = resp.model_info;
  for (unsigned int i = 0; i < resp.model_info.vocab.size(); i++) {
    self->vocab_[resp.model_info.vocab[i]] = i;
  }

  self->model_ = std::unique_ptr<LanguageModel>(
    new LanguageModel(resp.model_info));
  self->model_->set_params(resp.params);
  pthread_mutex_unlock(&self->model_lock_);

  pthread_mutex_lock(&self->shard_paths_lock_);
  for (const std::string &path : resp.shard_paths) {
    self->shard_paths_.push(path);
  }
  pthread_mutex_unlock(&self->shard_paths_lock_);

  self->learn_rate_ = resp.learn_rate;
  self->batch_size_ = resp.batch_size;

  // Signal start of computation
  self->stop_ = false;
  pthread_cond_signal(&self->stop_cond_);
  pthread_mutex_unlock(&self->stop_lock_);
  return NULL;
}

uint32_t
Worker::word_index(const std::string &word) {
  boost::regex re("[0-9]");
  std::string token = boost::regex_replace(word, re, "0");
  uint32_t index = model_info_.unk_token_index;
  auto itr = vocab_.find(token);
  if (itr != vocab_.end()) {
    index = itr->second;
  }
  return index;
}

void *
Worker::compute(void *arg) {
  Worker *self = (Worker *) arg;

  // Wait for computation to start
  pthread_mutex_lock(&self->stop_lock_);
  while (self->stop_) {
    pthread_cond_wait(&self->stop_cond_, &self->stop_lock_);
  }
  pthread_mutex_unlock(&self->stop_lock_);
  std::cout << "Starting computation" << std::endl;

  // Get shard path
  pthread_mutex_lock(&self->shard_paths_lock_);
  while (self->shard_paths_.size() == 0) {
    std::cout << "No shards assigned -- waiting" << std::endl;
    pthread_cond_wait(&self->has_shards_cond_, &self->shard_paths_lock_);
  }
  std::string cur_shard_path = self->shard_paths_.front();
  self->shard_paths_.pop();
  pthread_mutex_unlock(&self->shard_paths_lock_);
  std::ifstream cur_shard(cur_shard_path);

  int count = 0;

  // Compute gradient update
  while (true) {
    pthread_mutex_lock(&self->stop_lock_);
    while (self->stop_) {
      std::cout << "Stopped" << std::endl;
      pthread_cond_wait(&self->stop_cond_, &self->stop_lock_);
      std::cout << "Started" << std::endl;
    }
    pthread_mutex_unlock(&self->stop_lock_);

    // Read next line
    int window = self->model_info_.window_size;
    std::string line, word;
    std::vector<uint32_t> tokens;
    for (int i = 0; i < window - 1; i++) {
      tokens.push_back(self->model_info_.start_token_index);
    }

    while (!std::getline(cur_shard, line)) {
      pthread_mutex_lock(&self->completed_shards_lock_);
      self->completed_shards_.insert(cur_shard_path);
      pthread_mutex_unlock(&self->completed_shards_lock_);

      pthread_mutex_lock(&self->shard_paths_lock_);
      while (self->shard_paths_.size() == 0) {
        std::cout << "No more shards assigned -- waiting" << std::endl;
        pthread_cond_wait(&self->has_shards_cond_, &self->shard_paths_lock_);
      }
      cur_shard_path = self->shard_paths_.front();
      self->shard_paths_.pop();
      pthread_mutex_unlock(&self->shard_paths_lock_);
      cur_shard.close();
      cur_shard.open(cur_shard_path);
    }

    std::stringstream ss(line);
    
    while (std::getline(ss, word, ' ')) {
      tokens.push_back(self->word_index(word));
    }
    tokens.push_back(self->model_info_.end_token_index);

    // Compute gradient update
    std::cout << "Computing on sentence" << std::endl;
    for (unsigned int i = window; i < tokens.size(); i++) {
      uint32_t target = tokens[i];
      std::vector<uint32_t> input(
        tokens.begin() + i - window, tokens.begin() + i);
      self->model_->forward(input);
      self->model_->backward(input, target);
      count++;

      if (count == self->batch_size_) {
        count = 0;
        
        // get update and push
        std::cout << "Pushing update" << std::endl;
        ParamUpdate update;
        self->model_->get_update(update, self->learn_rate_);
        self->model_->zero_grad_params();
        self->param_client_->push_update(update);

        // pull latest parameters
        std::cout << "Pulling params" << std::endl;
        Params params;
        self->param_client_->pull_params(params);
        self->model_->set_params(params);
      }
    }
  }

  return NULL;
}

void
Worker::run() {
  int server_ret = pthread_create(&server_thread_, NULL, &Worker::server, this);
  int announce_ret = pthread_create(&announce_thread_, NULL, &Worker::announce, this);
  int compute_ret = pthread_create(&compute_thread_, NULL, &Worker::compute, this);
  pthread_join(server_thread_, NULL);
  pthread_join(announce_thread_, NULL);
  pthread_join(compute_thread_, NULL);
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

  Worker worker(master_ip, master_port, worker_port);
  worker.run();
  return 0;
}
