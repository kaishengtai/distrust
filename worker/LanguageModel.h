#ifndef LANGUAGEMODEL_H
#define LANGUAGEMODEL_H

#include <Eigen/Dense>

#include "../gen-cpp/distrust_types.h"

class LanguageModel {
public:
  LanguageModel(const distrust::ModelInfo& model_info);
  ~LanguageModel();

  void init(const distrust::Params& params);
  Eigen::VectorXd forward(Eigen::VectorXd input);
  //Eigen::MatrixXd forward(Eigen::MatrixXd input);
  distrust::Params backward(Eigen::VectorXd input);
  //distrust::Params backward(Eigen::MatrixXd input);

private:
  // Model information
  int window_size_;
  int wordvec_dim_;
  int hidden_dim_;
  int start_token_index_;
  int end_token_index_;
  int unk_token_index_;

  // Parameters
  Eigen::MatrixXd wordvecs_;
  Eigen::MatrixXd input_hidden_weights_;
  Eigen::VectorXd input_hidden_biases_;
  Eigen::MatrixXd hidden_output_weights_;
  Eigen::VectorXd hidden_output_biases_;
};

#endif
