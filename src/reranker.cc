#include <iostream>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/export.hpp>

#include "reranker.h"
BOOST_CLASS_EXPORT_IMPLEMENT(LinearRerankerModel)
BOOST_CLASS_EXPORT_IMPLEMENT(NonlinearRerankerModel)

using namespace std;

RerankerModel::RerankerModel() {
  num_dimensions = 0;
}

RerankerModel::RerankerModel(unsigned num_dimensions) : num_dimensions(num_dimensions) {}

RerankerModel::~RerankerModel() {}

Expression RerankerModel::BatchScore(vector<vector<float> >& features, vector<float>& gold_scores, ComputationGraph& cg) {
  assert (features.size() == gold_scores.size());
  vector<Expression> model_scores(features.size());
  for (unsigned i = 0; i < features.size(); ++i) {
    model_scores[i] = score(&features[i], cg);
  }
  Expression model_score_vector = concatenate(model_scores);
  return model_score_vector;
}

void RerankerModel::BuildComputationGraph(vector<vector<float> >& features, vector<float>& gold_scores, ComputationGraph& cg) {
  Expression model_score_vector = BatchScore(features, gold_scores, cg);
  Expression hyp_probs = softmax(model_score_vector);
  assert (features.size() < LONG_MAX);
  Expression ref_scores = input(cg, {(long)features.size()}, &gold_scores);
  Expression ebleu = dot_product(hyp_probs, ref_scores);
  Expression loss = -ebleu;
}

LinearRerankerModel::LinearRerankerModel() : RerankerModel(0) {
}

LinearRerankerModel::LinearRerankerModel(unsigned num_dimensions) : RerankerModel(num_dimensions) {
  InitializeParameters();
}

void LinearRerankerModel::InitializeParameters() {
  assert (num_dimensions > 0);
  p_w = cnn_model.add_parameters({1, num_dimensions});
}

Expression LinearRerankerModel::score(vector<float>* input_features, ComputationGraph& cg) {
  assert (num_dimensions > 0);
  Expression w = parameter(cg, p_w);
  Expression h = input(cg, {num_dimensions}, input_features);
  Expression s = w * h;
  return s;
}

NonlinearRerankerModel::NonlinearRerankerModel() : RerankerModel(0) {
  hidden_size = 0;
}

NonlinearRerankerModel::NonlinearRerankerModel(unsigned num_dimensions, unsigned hidden_layer_size) : RerankerModel(num_dimensions), hidden_size(hidden_layer_size) {
  InitializeParameters();
}

void NonlinearRerankerModel::InitializeParameters() {
  assert (num_dimensions > 0);
  p_w1 = cnn_model.add_parameters({hidden_size, num_dimensions});
  p_w2 = cnn_model.add_parameters({1, hidden_size});
  p_b = cnn_model.add_parameters({hidden_size});
}

Expression NonlinearRerankerModel::score(vector<float>* input_features, ComputationGraph& cg) {
  assert (num_dimensions > 0);
  Expression w1 = parameter(cg, p_w1);
  Expression w2 = parameter(cg, p_w2);
  Expression b = parameter(cg, p_b);
  Expression h = input(cg, {num_dimensions}, input_features);
  Expression g = affine_transform({b, w1, h});
  Expression t = tanh(g);
  Expression s = w2 * t;
  return s;
}
