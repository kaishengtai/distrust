#include "LanguageModel.h"

#include <iostream>
#include <math.h>
#include <random>
#include <sstream>

#include <boost/regex.hpp>

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
  for (unsigned int i = 0; i < vocab_size_; i++) {
    vocab_[model_info.vocab[i]] = i;
  }

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

  // adagrad
  for (unsigned int i = 0; i < vocab_size_; i++) {
    wordvec_w_var_.push_back(ArrayXd::Zero(wordvec_dim_));
  }

  for (unsigned int i = 0; i < window_size_; i++) {
    input_hidden_w_var_.push_back(MArray_t::Zero(hidden_dim_, wordvec_dim_));
  }

  input_hidden_b_var_ = ArrayXd::Zero(hidden_dim_);
  hidden_output_w_var_ = MArray_t::Zero(vocab_size_, hidden_dim_);
  hidden_output_b_var_ = ArrayXd::Zero(vocab_size_);
  zero_grad_params();
}

LanguageModel::LanguageModel(const LanguageModel &model) :
  window_size_(model.window_size_),
  wordvec_dim_(model.wordvec_dim_),
  hidden_dim_(model.hidden_dim_),
  start_token_index_(model.start_token_index_),
  end_token_index_(model.end_token_index_),
  unk_token_index_(model.unk_token_index_),
  vocab_size_(model.vocab_size_),
  vocab_(model.vocab_),
  wordvec_w_buf_(model.wordvec_w_buf_),
  input_hidden_w_buf_(model.input_hidden_w_buf_),
  input_hidden_b_buf_(model.input_hidden_b_buf_),
  hidden_output_w_buf_(model.hidden_output_w_buf_),
  hidden_output_b_buf_(model.hidden_output_b_buf_),
  input_hidden_w_grad_(model.input_hidden_w_grad_),
  input_hidden_b_grad_(model.input_hidden_b_grad_),
  hidden_output_w_grad_(model.hidden_output_w_grad_),
  hidden_output_b_grad_(model.hidden_output_b_grad_) {
  
  zero_grad_params();
  wrap_buffers();
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
  double *ptr;
  for (auto itr = wordvec_w_grad_.begin(); itr != wordvec_w_grad_.end(); itr++) {
    uint32_t idx = itr->first;
    //ArrayXd a = itr->second.array() * (-learn_rate);
    ArrayXd a = itr->second.array();
    wordvec_w_var_[idx] += a.square();
    a *= -learn_rate / (wordvec_w_var_[idx].sqrt() + 1e-4);
    ptr = a.data();
    ret.wordvec_w[idx] = std::vector<double>(ptr, ptr + wordvec_dim_);
  }

  for (unsigned int i = 0; i < window_size_; i++) {
    //MArray_t a = input_hidden_w_grad_[i].array() * (-learn_rate);
    MArray_t a = input_hidden_w_grad_[i].array();
    input_hidden_w_var_[i] += a.square();
    a *= -learn_rate / (input_hidden_w_var_[i].sqrt() + 1e-4);
    ptr = a.data();
    ret.input_hidden_w.push_back(std::vector<double>(ptr, ptr + hidden_dim_ * wordvec_dim_));
  }

  //ArrayXd a_ihb = input_hidden_b_grad_.array() * (-learn_rate);
  ArrayXd a_ihb = input_hidden_b_grad_.array();
  input_hidden_b_var_ += a_ihb.square();
  a_ihb *= -learn_rate / (input_hidden_b_var_.sqrt() + 1e-4);
  ptr = a_ihb.data();
  ret.input_hidden_b = std::vector<double>(ptr, ptr + hidden_dim_);

  //MArray_t a_how = hidden_output_w_grad_.array() * (-learn_rate);
  MArray_t a_how = hidden_output_w_grad_.array();
  hidden_output_w_var_ += a_how.square();
  a_how *= -learn_rate / (hidden_output_w_var_.sqrt() + 1e-4);
  ptr = a_how.data();
  ret.hidden_output_w = std::vector<double>(ptr, ptr + vocab_size_ * hidden_dim_);

  //ArrayXd a_hob = hidden_output_b_grad_.array() * (-learn_rate);
  ArrayXd a_hob = hidden_output_b_grad_.array();
  hidden_output_b_var_ += a_hob.square();
  a_hob *= -learn_rate / (hidden_output_b_var_.sqrt() + 1e-4);
  ptr = a_hob.data();
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

uint32_t
LanguageModel::word_index(const std::string &word) {
  boost::regex re("[0-9]");
  std::string token = boost::regex_replace(word, re, "0");
  uint32_t index = unk_token_index_;
  auto itr = vocab_.find(token);
  if (itr != vocab_.end()) {
    index = itr->second;
  }
  return index;
}

std::vector<uint32_t>
LanguageModel::tokenize(const std::string &line) {
  std::vector<uint32_t> tokens;
  for (unsigned int i = 0; i < window_size_; i++) {
    tokens.push_back(start_token_index_);
  }

  std::string word;
  std::stringstream ss(line);
  while (std::getline(ss, word, ' ')) {
    tokens.push_back(word_index(word));
  }
  tokens.push_back(end_token_index_);
  return tokens;
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

// double
// LanguageModel::forward(const std::vector<uint32_t> &tokens) {
//   uint32_t size = tokens.size() - window_size_;
//   Matrix<uint32_t, Dynamic, Dynamic> input(size, window_size_);
//   for (unsigned int i = 0; i < size; i++) {
//     for (unsigned int j = 0; j < window_size_; j++) {
//       input(i, j) = tokens[i + j];
//     }
//   }


// }

// double
// LanguageModel::forward(const std::vector<uint32_t> &input, const uint32_t target) {
//   if (input.size() != window_size_) {
//     throw std::invalid_argument("forward: input size does not equal window size");
//   }

//   hidden_ = *input_hidden_b_;
//   for (unsigned int i = 0; i < window_size_; i++) {
//     hidden_ += input_hidden_w_[i] * wordvec_w_[input[i]];
//   }
//   hidden_tanh_ = tanh(hidden_);
//   output_ = (*hidden_output_w_) * hidden_tanh_ + (*hidden_output_b_);
//   logZ_ = logZ(output_);
//   output_normed_ = (output_.array() - logZ_).matrix();
//   return output_normed_(target);
// }

// std::vector<double>
// LanguageModel::forward(const std::vector<uint32_t> &input, uint32_t target_idx) {
//   hidden_ = *input_hidden_b_;
//   for (unsigned int i = 0; i < window_size_; i++) {
//     hidden_ += input_hidden_w_[i] * wordvec_w_[input[target_idx - window_size_ + i]];
//   }
//   hidden_tanh_ = tanh(hidden_);
//   output_ = (*hidden_output_w_) * hidden_tanh_ + (*hidden_output_b_);
//   logZ_ = logZ(output_);
//   output_normed_ = (output_.array() - logZ_).matrix();

//   double *ptr = output_normed_.data();
//   return std::vector<double>(ptr, ptr + vocab_size_);
// }

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
