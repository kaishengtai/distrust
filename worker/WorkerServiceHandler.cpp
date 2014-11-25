#include "WorkerServiceHandler.h"

#include <iostream>

using boost::shared_ptr;

using namespace distrust;

void
WorkerServiceHandler::heartbeat(HBResponse& _return) {
  std::cout << "heartbeat" << std::endl;
}

void
WorkerServiceHandler::start(const StartRequest& request) {
  std::cout << "start" << std::endl;
}

void
WorkerServiceHandler::stop() {
  std::cout << "stop" << std::endl;
}

void
WorkerServiceHandler::reassign(const std::vector<std::string> & shard_paths) {
  std::cout << "reassign" << std::endl;
}
