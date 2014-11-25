#include <iostream>
#include <getopt.h>
#include <pthread.h>

#include "../gen-cpp/ParamService.h"
#include "LanguageModel.h"
#include "WorkerServiceHandler.h"

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

class Worker {
public:
  Worker(const std::string& master_ip, const int master_port, const int worker_port) :
    master_ip_(master_ip),
    master_port_(master_port),
    worker_port_(worker_port),
    server_ready_(false) {

    pthread_mutex_init(&lock_, NULL);
    pthread_cond_init(&server_ready_cond_, NULL);

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

  ~Worker() {}

  static void *server(void *arg) {
    Worker *context = (Worker *) arg;
    pthread_mutex_lock(&context->lock_);
    const int worker_port = context->worker_port_;
    std::cout << "Starting worker on port " << worker_port << std::endl;
    shared_ptr<WorkerServiceHandler> handler(new WorkerServiceHandler());
    shared_ptr<TProcessor> processor(new WorkerServiceProcessor(handler));
    shared_ptr<TServerTransport> serverTransport(new TServerSocket(worker_port));
    shared_ptr<TTransportFactory> transportFactory(new TBufferedTransportFactory());
    shared_ptr<TProtocolFactory> protocolFactory(new TBinaryProtocolFactory());
    TThreadedServer server(processor, serverTransport, transportFactory, protocolFactory);
    context->server_ready_ = true;
    pthread_cond_signal(&context->server_ready_cond_);
    pthread_mutex_unlock(&context->lock_);
    server.serve();
    return NULL;
  }

  static void *announce(void *arg) {
    Worker *context = (Worker *) arg;

    pthread_mutex_lock(&context->lock_);
    while (!context->server_ready_) {
      pthread_cond_wait(&context->server_ready_cond_, &context->lock_);
    }
    pthread_mutex_unlock(&context->lock_);

    // give the server time to start up
    sleep(1);

    AnnounceResponse resp;
    context->param_client_->announce(resp, context->worker_port_);
    context->model_ = std::unique_ptr<LanguageModel>(
      new LanguageModel(resp.model_info));
    context->model_->init(resp.params);
    context->shard_paths_ = resp.shard_paths;
    context->learn_rate_ = resp.learn_rate;

    std::cout << "Received paths from master:" << std::endl;
    for (const std::string& path : context->shard_paths_) {
      std::cout << path << std::endl;
    }

    return NULL;
  }

  void run() {
    int server_ret = pthread_create(&server_thread_, NULL, &Worker::server, this);
    int announce_ret = pthread_create(&announce_thread_, NULL, &Worker::announce, this);
    pthread_join(server_thread_, NULL);
    pthread_join(announce_thread_, NULL);
  }

protected:
  std::unique_ptr<ParamServiceClient> param_client_;
  std::unique_ptr<LanguageModel> model_;

  // Paths to training data
  std::vector<std::string> shard_paths_;

  // Learning rate
  double learn_rate_;

  pthread_t server_thread_, announce_thread_, pull_thread_, push_thread_;
  std::string master_ip_;
  int master_port_, worker_port_;

  // server/client synchronization
  pthread_mutex_t lock_;
  bool server_ready_;
  pthread_cond_t server_ready_cond_;
};

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
