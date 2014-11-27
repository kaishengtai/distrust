#include "Worker.h"

#include <iostream>
#include <getopt.h>
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

Worker::Worker(const std::string& master_ip, const int master_port, const int worker_port) :
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
  Worker *context = (Worker *) arg;
  pthread_mutex_lock(&context->server_ready_lock_);
  const int worker_port = context->worker_port_;
  std::cout << "Starting worker on port " << worker_port << std::endl;
  shared_ptr<WorkerServiceHandler> handler(new WorkerServiceHandler(context));
  shared_ptr<TProcessor> processor(new WorkerServiceProcessor(handler));
  shared_ptr<TServerTransport> serverTransport(new TServerSocket(worker_port));
  shared_ptr<TTransportFactory> transportFactory(new TBufferedTransportFactory());
  shared_ptr<TProtocolFactory> protocolFactory(new TBinaryProtocolFactory());
  TThreadedServer server(processor, serverTransport, transportFactory, protocolFactory);
  context->server_ready_ = true;
  pthread_cond_signal(&context->server_ready_cond_);
  pthread_mutex_unlock(&context->server_ready_lock_);
  server.serve();
  return NULL;
}

void *
Worker::announce(void *arg) {
  Worker *context = (Worker *) arg;

  // Wait until the server signals that it's starting up.
  pthread_mutex_lock(&context->server_ready_lock_);
  while (!context->server_ready_) {
    pthread_cond_wait(&context->server_ready_cond_, &context->server_ready_lock_);
  }
  pthread_mutex_unlock(&context->server_ready_lock_);

  // Give the server time to start up
  sleep(1);

  // Announce the worker to the master.
  // No requests from the master will be received until after this point.
  pthread_mutex_lock(&context->stop_lock_);
  AnnounceResponse resp;
  context->param_client_->announce(resp, context->worker_port_);

  // Initialize model parameters.
  // Takes ownership of memory holding parameters.
  pthread_mutex_lock(&context->model_lock_);
  context->model_ = std::unique_ptr<LanguageModel>(
    new LanguageModel(resp.model_info));
  context->model_->set_params(resp.params);
  pthread_mutex_unlock(&context->model_lock_);

  pthread_mutex_lock(&context->shard_paths_lock_);
  context->shard_paths_ = resp.shard_paths;
  pthread_mutex_unlock(&context->shard_paths_lock_);

  context->learn_rate_ = resp.learn_rate;

  // Signal start of computation
  context->stop_ = false;
  pthread_cond_signal(&context->stop_cond_);
  pthread_mutex_unlock(&context->stop_lock_);
  return NULL;
}

void *
Worker::compute(void *arg) {
  Worker *context = (Worker *) arg;

  while (true) {
    pthread_mutex_lock(&context->stop_lock_);
    while (context->stop_) {
      std::cout << "Stopped" << std::endl;
      pthread_cond_wait(&context->stop_cond_, &context->stop_lock_);
      std::cout << "Started" << std::endl;
    }
    pthread_mutex_unlock(&context->stop_lock_);

    // Perform computation on a batch
    std::cout << "Computing on batch" << std::endl;
    sleep(4);
  }

  return NULL;
}

void *
Worker::push(void *arg) {
  Worker *context = (Worker *) arg;

  while (true) {
    sleep(5);
    std::cout << "Pushing update" << std::endl;
  }

  return NULL;
}

void *
Worker::pull(void *arg) {
  Worker *context = (Worker *) arg;

  while (true) {
    sleep(5);
    std::cout << "Pulling parameters" << std::endl;
  }

  return NULL;
}

void
Worker::run() {
  int server_ret = pthread_create(&server_thread_, NULL, &Worker::server, this);
  int announce_ret = pthread_create(&announce_thread_, NULL, &Worker::announce, this);
  int compute_ret = pthread_create(&compute_thread_, NULL, &Worker::compute, this);
  int push_ret = pthread_create(&push_thread_, NULL, &Worker::push, this);
  int pull_ret = pthread_create(&pull_thread_, NULL, &Worker::pull, this);
  pthread_join(server_thread_, NULL);
  pthread_join(announce_thread_, NULL);
  pthread_join(compute_thread_, NULL);
  pthread_join(push_thread_, NULL);
  pthread_join(pull_thread_, NULL);
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
