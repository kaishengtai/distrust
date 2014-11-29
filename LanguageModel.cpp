#include "LanguageModel.h"

#include <math.h>
#include <random>

using namespace distrust;
using namespace Eigen;

LanguageModel::LanguageModel(const ModelInfo &model_info) :
  unif_(-1.0, 1.0),
  batch_size_(0) {

  window_size_ = model_info.window_size;
  wordvec_dim_ = model_info.wordvec_dim;
  hidden_dim_ = model_info.hidden_dim;
  start_token_index_ = model_info.start_token_index;
  end_token_index_ = model_info.end_token_index;
  unk_token_index_ = model_info.unk_token_index;
  vocab_size_ = model_info.vocab.size();

  wordvec_w_.reserve(vocab_size_);
  wordvec_w_buf_.reserve(vocab_size_);
  input_hidden_w_.reserve(window_size_);
  input_hidden_w_buf_.reserve(window_size_);

  for (unsigned int i = 0; i < window_size_; i++) {
    input_hidden_w_grad_.push_back(Matrix_t(hidden_dim_, wordvec_dim_));
  }
  input_hidden_b_grad_ = Vector_t(hidden_dim_);
  hidden_output_w_grad_ = Matrix_t(vocab_size_, hidden_dim_);
  hidden_output_b_grad_ = Vector_t(vocab_size_);
  zero_grad_params();
}

double
LanguageModel::sample() {
  return unif_(re_);
}

void
LanguageModel::wrap_buffers() {
  wordvec_w_.clear();
  for (unsigned int i = 0; i < vocab_size_; i++) {
    wordvec_w_.push_back(Map<Vector_t>(&wordvec_w_buf_[i][0], wordvec_dim_));
  } 

  input_hidden_w_.clear();
  for (unsigned int i = 0; i < window_size_; i++) {
    input_hidden_w_.push_back(Map<Matrix_t>(&input_hidden_w_buf_[i][0], hidden_dim_, wordvec_dim_));
  }

  input_hidden_b_ = std::unique_ptr<Map<Vector_t>>(
    new Map<Vector_t>(&input_hidden_b_buf_[0], hidden_dim_));
  hidden_output_w_ = std::unique_ptr<Map<Matrix_t>>(
    new Map<Matrix_t>(&hidden_output_w_buf_[0], vocab_size_, hidden_dim_));
  hidden_output_b_ = std::unique_ptr<Map<Vector_t>>(
    new Map<Vector_t>(&hidden_output_b_buf_[0], vocab_size_));
}

void
LanguageModel::random_init() {
  wordvec_w_buf_.clear();
  for (unsigned int i = 0; i < vocab_size_; i++) {
    wordvec_w_buf_.push_back(std::vector<double>(wordvec_dim_));
    for (unsigned int j = 0; j < wordvec_dim_; j++) {
      wordvec_w_buf_[i][j] = 0.05 * sample();
    }
  }

  input_hidden_w_buf_.clear();
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
  for (auto itr = update.wordvec_w.begin(); itr != update.wordvec_w.end(); itr++) {
    uint32_t idx = itr->first;
    for (unsigned int i = 0; i < wordvec_dim_; i++) {
      wordvec_w_buf_[idx][i] += itr->second[i];
    }
  }

  for (unsigned int i = 0; i < window_size_; i++) {
    for (unsigned int j = 0; j < wordvec_dim_ * hidden_dim_; j++) {
      input_hidden_w_buf_[i][j] += update.input_hidden_w[i][j];
    }
  }

  for (unsigned int i = 0; i < hidden_dim_; i++) {
    input_hidden_b_buf_[i] += update.input_hidden_b[i];
  }

  for (unsigned int i = 0; i < hidden_dim_ * vocab_size_; i++) {
    hidden_output_w_buf_[i] += update.hidden_output_w[i];
  }

  for (unsigned int i = 0; i < vocab_size_; i++) {
    hidden_output_b_buf_[i] += update.hidden_output_b[i];
  }
}

void
LanguageModel::get_params(Params &ret) {
  ret.wordvec_w = wordvec_w_buf_;
  ret.input_hidden_w = input_hidden_w_buf_;
  ret.input_hidden_b = input_hidden_b_buf_;
  ret.hidden_output_w = hidden_output_w_buf_;
  ret.hidden_output_b = hidden_output_b_buf_;
}

void
LanguageModel::get_update(ParamUpdate &ret, const double learn_rate) {
  ArrayXd a;
  double *ptr;
  for (auto itr = wordvec_w_grad_.begin(); itr != wordvec_w_grad_.end(); itr++) {
    uint32_t idx = itr->first;
    a = itr->second.array() * learn_rate / batch_size_;
    ptr = a.data();
    ret.wordvec_w[idx] = std::vector<double>(ptr, ptr + wordvec_dim_);
  }

  for (unsigned int i = 0; i < window_size_; i++) {
    a = input_hidden_w_grad_[i].array() * learn_rate / batch_size_;
    ptr = a.data();
    ret.input_hidden_w.push_back(std::vector<double>(ptr, ptr + hidden_dim_ * wordvec_dim_));
  }

  a = input_hidden_b_grad_.array() * learn_rate / batch_size_;
  ptr = a.data();
  ret.input_hidden_b = std::vector<double>(ptr, ptr + hidden_dim_);

  a = hidden_output_w_grad_.array() * learn_rate / batch_size_;
  ptr = a.data();
  ret.hidden_output_w = std::vector<double>(ptr, ptr + vocab_size_ * hidden_dim_);

  a = hidden_output_b_grad_.array() * learn_rate / batch_size_;
  ptr = a.data();
  ret.hidden_output_b = std::vector<double>(ptr, ptr + vocab_size_);
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

  double *ptr = output_normed_.data();
  return std::vector<double>(ptr, ptr + vocab_size_);
}

void
LanguageModel::backward(
  const std::vector<uint32_t> &input,
  const uint32_t target) {

  batch_size_++;

  // hidden-output gradients
  Vector_t output_grad = output_normed_.array().exp().matrix();
  output_grad(target) -= 1;
  hidden_output_w_grad_ += output_grad * hidden_tanh_.transpose();
  hidden_output_b_grad_ += output_grad;

  // input-hidden gradients
  Vector_t hidden_grad = output_grad.transpose() * (*hidden_output_w_);
  hidden_grad = (hidden_grad.array() * 
    (1 - hidden_tanh_.array().square()))
    .matrix();
  for (unsigned int i = 0; i < window_size_; i++) {
    input_hidden_w_grad_[i] += hidden_grad * wordvec_w_[input[i]].transpose();
  }
  input_hidden_b_grad_ += hidden_grad;

  // word vector gradients
  for (unsigned int i = 0; i < window_size_; i++) {
    uint32_t idx = input[i];
    Vector_t input_grad = hidden_grad.transpose() * input_hidden_w_[i];
    auto itr = wordvec_w_grad_.find(idx);
    if (itr == wordvec_w_grad_.end()) {
      wordvec_w_grad_[idx] = input_grad;
    } else {
      wordvec_w_grad_[idx] += input_grad;
    }
  }
}

void
LanguageModel::zero_grad_params() {
  batch_size_ = 0;
  wordvec_w_grad_.clear();
  for (unsigned int i = 0; i < window_size_; i++) {
    input_hidden_w_grad_[i].setZero();
  }
  input_hidden_b_grad_.setZero();
  hidden_output_w_grad_.setZero();
  hidden_output_b_grad_.setZero();
}
