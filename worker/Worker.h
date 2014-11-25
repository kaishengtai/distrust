#include <Eigen/Dense>

#include "../gen-cpp/WorkerService.h"
#include "../gen-cpp/ParamService.h"

using namespace distrust;

#define WORKER_DEFAULT_PORT 9090

class WorkerServiceHandler : virtual public WorkerServiceIf {

public:
  WorkerServiceHandler(const std::string& master_ip, int master_port);
  ~WorkerServiceHandler();

  void heartbeat(HBResponse& _return);
  void start(const StartRequest& request);
  void stop();
  void reassign(const std::vector<std::string> & shard_paths);

private:

private:
  std::unique_ptr<ParamServiceClient> param_client_;

  // Parameter server info
  std::string master_ip_;
  int master_port_;

  // Paths to training data
  std::vector<std::string> shard_paths_;

  // Learning rate
  double learn_rate_;

  // Model information
  int window_size_;
  int wordvec_dim_;
  int hidden_dim_;
  int vocab_size_;
  int start_token_index_;
  int end_token_index_;

  // Parameters
  Eigen::MatrixXd wordvecs_;
  Eigen::MatrixXd input_hidden_weights_;
  Eigen::VectorXd input_hidden_biases_;
  Eigen::MatrixXd hidden_output_weights_;
  Eigen::VectorXd hidden_output_biases_;
};
