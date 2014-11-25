#include "LanguageModel.h"

using namespace distrust;
using namespace Eigen;

LanguageModel::LanguageModel(const ModelInfo& model_info) {

}

LanguageModel::~LanguageModel() {

}

void
LanguageModel::init(const distrust::Params& params) {

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
