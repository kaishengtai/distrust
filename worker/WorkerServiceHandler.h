#ifndef WORKERSERVICEHANDLER_H
#define WORKERSERVICEHANDLER_H

#include "../gen-cpp/WorkerService.h"

using namespace distrust;

#define WORKER_DEFAULT_PORT 9090

class WorkerServiceHandler : virtual public WorkerServiceIf {

public:
  WorkerServiceHandler() { }
  ~WorkerServiceHandler() { }

  void heartbeat(HBResponse& _return);
  void start(const StartRequest& request);
  void stop();
  void reassign(const std::vector<std::string> & shard_paths);

private:

};

#endif
