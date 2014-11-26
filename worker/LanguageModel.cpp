#include "LanguageModel.h"

using namespace distrust;
using namespace Eigen;

LanguageModel::LanguageModel(const ModelInfo &model_info) {
  window_size_ = model_info.window_size;
  wordvec_dim_ = model_info.wordvec_dim;
  hidden_dim_ = model_info.hidden_dim;
  vocab_size_ = model_info.vocab_size;
  start_token_index_ = model_info.start_token_index;
  end_token_index_ = model_info.end_token_index;
}

LanguageModel::~LanguageModel() {

}

void
LanguageModel::init(const distrust::Params &params) {

}

VectorXd
forward(VectorXd input) {
  return VectorXd(1);
}

Params
backward(VectorXd input) {
  Params params;
  return params;
}
