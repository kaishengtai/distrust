#include "LanguageModel.h"

#include <math.h>
#include <random>

using namespace distrust;
using namespace Eigen;

LanguageModel::LanguageModel(const ModelInfo &model_info) :
  unif_(-1.0, 1.0) {

  window_size_ = model_info.window_size;
  wordvec_dim_ = model_info.wordvec_dim;
  hidden_dim_ = model_info.hidden_dim;
  start_token_index_ = model_info.start_token_index;
  end_token_index_ = model_info.end_token_index;
  unk_token_index_ = model_info.unk_token_index;
  vocab_size_ = model_info.vocab.size();
}

double
LanguageModel::sample() {
  return unif_(re_);
}

void
LanguageModel::wrap_buffers() {
  for (unsigned int i = 0; i < vocab_size_; i++) {
    wordvec_w_.push_back(Vector_t(&wordvec_w_buf_[i][0], wordvec_dim_));
  } 

  for (unsigned int i = 0; i < window_size_; i++) {
    input_hidden_w_.push_back(Matrix_t(&input_hidden_w_buf_[i][0], hidden_dim_, wordvec_dim_));
  }

  input_hidden_b_ = std::unique_ptr<Vector_t>(new Vector_t(&input_hidden_b_buf_[0], hidden_dim_));
  hidden_output_w_ = std::unique_ptr<Matrix_t>(new Matrix_t(&hidden_output_w_buf_[0], vocab_size_, hidden_dim_));
  hidden_output_b_ = std::unique_ptr<Vector_t>(new Vector_t(&hidden_output_b_buf_[0], vocab_size_));
}

void
LanguageModel::random_init() {
  for (unsigned int i = 0; i < vocab_size_; i++) {
    wordvec_w_buf_.push_back(std::vector<double>(wordvec_dim_));
    for (unsigned int j = 0; j < wordvec_dim_; j++) {
      wordvec_w_buf_[i][j] = 0.05 * sample();
    }
  }

  for (unsigned int i = 0; i < window_size_; i++) {
    input_hidden_w_buf_.push_back(std::vector<double>(hidden_dim_ * wordvec_dim_));
    for (unsigned int j = 0; j < hidden_dim_ * wordvec_dim_; j++) {
      input_hidden_w_buf_[i][j] = 0.05 * sample();
    }
  }

  input_hidden_b_buf_ = std::vector<double>(hidden_dim_, 0.0);
  hidden_output_w_buf_ = std::vector<double>(vocab_size_ * hidden_dim_, 0.0);
  hidden_output_b_buf_ = std::vector<double>(vocab_size_, 0.0);
  wrap_buffers();
}

void
LanguageModel::set_params(Params &params) {
  wordvec_w_buf_ = std::move(params.wordvec_w);
  input_hidden_w_buf_ = std::move(params.input_hidden_w);
  input_hidden_b_buf_ = std::move(params.input_hidden_b);
  hidden_output_w_buf_ = std::move(params.hidden_output_w);
  hidden_output_b_buf_ = std::move(params.hidden_output_b);
  wrap_buffers();
}

void
LanguageModel::update_params(const ParamUpdate &update) {

}

void
LanguageModel::get_params(Params &ret) {
  ret.wordvec_w = wordvec_w_buf_;
  ret.input_hidden_w = input_hidden_w_buf_;
  ret.input_hidden_b = input_hidden_b_buf_;
  ret.hidden_output_w = hidden_output_w_buf_;
  ret.hidden_output_b = hidden_output_b_buf_;
}

VectorXd
LanguageModel::tanh(const VectorXd &v) {
  ArrayXd a = (v.array() * 2).exp();
  return ((a - 1) / (a + 1)).matrix();
}

double
LanguageModel::logZ(const VectorXd &v) {
  return log(v.array().exp().sum());
}

std::vector<double>
LanguageModel::forward(const std::vector<uint32_t> &input) {
  if (input.size() != window_size_) {
    throw std::invalid_argument("forward: input size does not equal window size");
  }

  hidden_ = *input_hidden_b_;
  for (unsigned int i = 0; i < window_size_; i++) {
    hidden_ += input_hidden_w_[i] * wordvec_w_[input[i]];
  }
  hidden_tanh_ = tanh(hidden_);
  output_ = (*hidden_output_w_) * hidden_tanh_ + (*hidden_output_b_);
  logZ_ = logZ(output_);
  output_normed_ = (output_.array() - logZ_).matrix();

  std::vector<double> result(vocab_size_);
  for (unsigned int i = 0; i < vocab_size_; i++) {
    result[i] = output_normed_(i);
  }
  return result;
}

void
LanguageModel::backward(
  Params &ret,
  const std::vector<uint32_t> &input,
  const uint32_t target) {


}
