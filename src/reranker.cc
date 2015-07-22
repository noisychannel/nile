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

RerankerModel::RerankerModel(Model* cnn_model, unsigned num_dimensions) : cnn_model(cnn_model), num_dimensions(num_dimensions) {}

RerankerModel::~RerankerModel() {}

Expression RerankerModel::BatchScore(const vector<Expression>& features, ComputationGraph& cg) {
  assert (features.size() > 0);
  vector<Expression> model_scores(features.size());
  for (unsigned i = 0; i < features.size(); ++i) {
    model_scores[i] = score(features[i], cg);
  }
  Expression model_score_vector = concatenate(model_scores);
  return model_score_vector;
}

void RerankerModel::BuildComputationGraph(const vector<Expression>& features, const vector<Expression>& metric_scores, ComputationGraph& cg) {
  Expression model_score_vector = BatchScore(features, cg);
  Expression hyp_probs = softmax(model_score_vector);
  assert (features.size() < LONG_MAX);
  Expression gold_scores = concatenate(metric_scores);
  Expression ebleu = dot_product(hyp_probs, gold_scores);
  Expression loss = -ebleu;
}

LinearRerankerModel::LinearRerankerModel() : RerankerModel(NULL, 0) {
}

LinearRerankerModel::LinearRerankerModel(Model* cnn_model, unsigned num_dimensions) : RerankerModel(cnn_model, num_dimensions) {
  InitializeParameters();
}

void LinearRerankerModel::InitializeParameters() {
  assert (num_dimensions > 0);
  p_w = cnn_model->add_parameters({1, num_dimensions});
}

Expression LinearRerankerModel::score(Expression h, ComputationGraph& cg) {
  Expression w = parameter(cg, p_w);
  Expression s = w * h;
  return s;
}

NonlinearRerankerModel::NonlinearRerankerModel() : RerankerModel(NULL, 0) {
  hidden_size = 0;
}

NonlinearRerankerModel::NonlinearRerankerModel(Model* cnn_model, unsigned num_dimensions, unsigned hidden_layer_size) : RerankerModel(cnn_model, num_dimensions), hidden_size(hidden_layer_size) {
  InitializeParameters();
}

void NonlinearRerankerModel::InitializeParameters() {
  assert (num_dimensions > 0);
  p_w1 = cnn_model->add_parameters({hidden_size, num_dimensions});
  p_w2 = cnn_model->add_parameters({1, hidden_size});
  p_b = cnn_model->add_parameters({hidden_size});
}

Expression NonlinearRerankerModel::score(Expression h, ComputationGraph& cg) {
  Expression w1 = parameter(cg, p_w1);
  Expression w2 = parameter(cg, p_w2);
  Expression b = parameter(cg, p_b);
  Expression g = affine_transform({b, w1, h});
  Expression t = tanh(g);
  Expression s = w2 * t;
  return s;
}
