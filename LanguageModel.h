#ifndef LANGUAGEMODEL_H
#define LANGUAGEMODEL_H

#include <Eigen/Dense>

#include "gen-cpp/distrust_types.h"

using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;

typedef Map<Matrix<double, Dynamic, Dynamic, RowMajor>> Matrix_t;
typedef Map<Matrix<double, Dynamic, 1>> Vector_t;

class LanguageModel {
public:
  LanguageModel(const distrust::ModelInfo &model_info);
  ~LanguageModel() {};

  void random_init();
  void set_params(distrust::Params &params);
  VectorXd forward(const VectorXd &input);
  //Eigen::MatrixXd forward(Eigen::MatrixXd input);
  distrust::Params backward(const VectorXd &input, const int32_t target);
  //distrust::Params backward(Eigen::MatrixXd input);

private:
  // Model information
  int32_t window_size_;
  int32_t wordvec_dim_;
  int32_t hidden_dim_;
  int32_t start_token_index_;
  int32_t end_token_index_;
  int32_t unk_token_index_;
  int32_t vocab_size_; 

  // Parameters
  std::vector<std::vector<double>> wordvec_w_buf_;
  std::vector<std::vector<double>> input_hidden_w_buf_;
  std::vector<double> input_hidden_b_buf_;
  std::vector<double> hidden_output_w_buf_;
  std::vector<double> hidden_output_b_buf_;
  std::vector<Vector_t> wordvec_w_;
  std::vector<Matrix_t> input_hidden_w_;
  std::unique_ptr<Vector_t> input_hidden_b_;
  std::unique_ptr<Matrix_t> hidden_output_w_;
  std::unique_ptr<Vector_t> hidden_output_b_;
};

#endif
