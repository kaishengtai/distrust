#include "LanguageModel.h"

using namespace distrust;
using namespace Eigen;

LanguageModel::LanguageModel(const ModelInfo &model_info) {
  window_size_ = model_info.window_size;
  wordvec_dim_ = model_info.wordvec_dim;
  hidden_dim_ = model_info.hidden_dim;
  start_token_index_ = model_info.start_token_index;
  end_token_index_ = model_info.end_token_index;
  unk_token_index_ = model_info.unk_token_index;
  vocab_size_ = model_info.vocab.size();
}

void
LanguageModel::random_init() {
  
}

void
LanguageModel::set_params(distrust::Params &params) {
  wordvec_w_buf_ = std::move(params.wordvec_w);
  //vector<double> wordvec_weights = &params.wordvec_weights[0];
  // double *input_hidden_weights = &params.input_hidden_weights[0];
  // double *input_hidden_biases = &params.input_hidden_biases[0];
  // double *hidden_output_weights = &params.hidden_output_weights[0];
  // double *hidden_output_biases = &params.hidden_output_biases[0];

  for (int i = 0; i < vocab_size_; i++) {
    wordvec_w_[i] = Vector_t(&wordvec_w_buf_[i][0], wordvec_dim_);
  }
}

VectorXd
forward(const VectorXd &input) {
  return VectorXd(1);
}

Params
backward(const VectorXd &input, const int32_t target) {
  Params params;
  return params;
}
