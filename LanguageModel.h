#ifndef LANGUAGEMODEL_H
#define LANGUAGEMODEL_H

#include <unordered_map>
#include <Eigen/Dense>

#include "gen-cpp/distrust_types.h"

using Eigen::Array;
using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::Matrix;
using Eigen::Map;
using Eigen::VectorXd;

typedef Matrix<double, Dynamic, Dynamic, RowMajor> Matrix_t;
typedef Array<double, Dynamic, Dynamic, RowMajor> MArray_t;
typedef Matrix<double, Dynamic, 1> Vector_t;

class LanguageModel {
 public:
  LanguageModel(const distrust::ModelInfo &model_info);
  ~LanguageModel() {};

  void random_init();
  void set_params(distrust::Params &params);
  void update_params(const distrust::ParamUpdate &update);
  void get_params(distrust::Params &ret);
  void get_update(distrust::ParamUpdate &ret, const double learn_rate);
  std::vector<double> forward(const std::vector<uint32_t> &input);
  //Eigen::MatrixXd forward(Eigen::MatrixXd input);
  void backward(const std::vector<uint32_t> &input, const uint32_t target);
  //distrust::Params backward(Eigen::MatrixXd input);
  void zero_grad_params();

 protected:
  inline double sample();
  void wrap_buffers();
  Eigen::VectorXd tanh(const Eigen::VectorXd &v);
  double logZ(const Eigen::VectorXd &v);

 protected:
  // Model information
  uint32_t window_size_;
  uint32_t wordvec_dim_;
  uint32_t hidden_dim_;
  uint32_t start_token_index_;
  uint32_t end_token_index_;
  uint32_t unk_token_index_;
  uint32_t vocab_size_; 

  // Source of randomness
  std::uniform_real_distribution<double> unif_;
  std::default_random_engine re_;

  // Parameters
  std::vector<std::vector<double>> wordvec_w_buf_;
  std::vector<std::vector<double>> input_hidden_w_buf_;
  std::vector<double> input_hidden_b_buf_;
  std::vector<double> hidden_output_w_buf_;
  std::vector<double> hidden_output_b_buf_;
  std::vector<Map<Vector_t>> wordvec_w_;
  std::vector<Map<Matrix_t>> input_hidden_w_;
  std::unique_ptr<Map<Vector_t>> input_hidden_b_;
  std::unique_ptr<Map<Matrix_t>> hidden_output_w_;
  std::unique_ptr<Map<Vector_t>> hidden_output_b_;

  // Cached activations
  Vector_t hidden_, hidden_tanh_;
  Vector_t output_, output_normed_;

  // Log partition function
  double logZ_;

  // Gradients
  std::unordered_map<uint32_t, Vector_t> wordvec_w_grad_;
  std::vector<Matrix_t> input_hidden_w_grad_;
  Vector_t input_hidden_b_grad_;
  Matrix_t hidden_output_w_grad_;
  Vector_t hidden_output_b_grad_;

  // Batch size counter
  uint32_t batch_size_;
};

#endif
