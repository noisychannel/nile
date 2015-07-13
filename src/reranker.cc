#include <iostream>
#include <fstream>
#include <algorithm>
#include <climits>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/export.hpp>

#include "reranker.h"
BOOST_CLASS_EXPORT_IMPLEMENT(LinearRerankerModel)
BOOST_CLASS_EXPORT_IMPLEMENT(NonlinearRerankerModel)

using namespace std;

// Takes the name of a kbest file name and returns the
// number of unique feature names found therewithin
unordered_set<string> get_feature_names(string filename, unsigned max_size) {
  unordered_set<string> feature_names;
  if (max_size == 0) {
    return feature_names;
  }

  ifstream input_stream(filename);
  for (string line; getline(input_stream, line);) {
    KbestHypothesis hyp = KbestHypothesis::parse(line);
    map<string, double>& features = hyp.features;
    for (map<string, double>::iterator it = features.begin(); it != features.end(); ++it) {
      feature_names.insert(it->first);
      if (feature_names.size() >= max_size) {
        input_stream.close();
        return feature_names;
      }
    }
    cerr << hyp.sentence_id << "\r";
  }
  input_stream.close();
  return feature_names;
}

unordered_set<string> get_feature_names(string filename) {
  return get_feature_names(filename, UINT_MAX);
}


RerankerModel::RerankerModel() {
  num_dimensions = 0;
}

RerankerModel::~RerankerModel() {}

void RerankerModel::ReadFeatureNames(string kbest_filename) {
  ReadFeatureNames(kbest_filename, UINT_MAX);
}

void RerankerModel::ReadFeatureNames(string kbest_filename, unsigned max_features) {
  cerr << "Reading feature names from k-best list...\n";
  unordered_set<string> feature_names = get_feature_names(kbest_filename, max_features);
  num_dimensions = feature_names.size();
  assert (num_dimensions > 0);
  cerr << "Found " << num_dimensions << " features.\n";

  cerr << "Building feature name-id maps...\n";
  unsigned feat_map_index = feat2id.size();
  for (string name : feature_names) {
    feat2id[name] = feat_map_index;
    id2feat[feat_map_index] = name;
    feat_map_index++;
  }
}

void RerankerModel::ConvertFeatureVector(KbestHypothesis& hypothesis, vector<float>& out) {
  assert (num_dimensions > 0);
  assert (out.size() == num_dimensions);
  fill(out.begin(), out.end(), 0.0);
  for (auto& kvp : hypothesis.features) {
    unsigned feat_id = feat2id[kvp.first];
    assert (feat_id < num_dimensions);
    out[feat_id] = kvp.second;
  }
}

void RerankerModel::ConvertKbestSet(vector<KbestHypothesis>& hyps, vector<vector<float> >& features, vector<float>& scores) {
  assert (num_dimensions > 0);
  features.resize(hyps.size());
  scores.resize(hyps.size());
  for (unsigned i = 0; i < hyps.size(); ++i) {
    features[i].resize(num_dimensions);
    ConvertFeatureVector(hyps[i], features[i]);
    scores[i] = hyps[i].metric_score;
  }
}

void RerankerModel::BuildComputationGraph(vector<vector<float> >& features, vector<float>& gold_scores, ComputationGraph& cg) {
  assert (features.size() == gold_scores.size());
  vector<Expression> model_scores(features.size());
  for (unsigned i = 0; i < features.size(); ++i) {
    model_scores[i] = score(&features[i], cg);
  }
  Expression model_score_vector = concatenate(model_scores);
  Expression hyp_probs = softmax(model_score_vector);
  assert (features.size() < LONG_MAX);
  Expression ref_scores = input(cg, {(long)features.size()}, &gold_scores);
  Expression ebleu = dot_product(hyp_probs, ref_scores);
  Expression loss = -ebleu;
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

NonlinearRerankerModel::NonlinearRerankerModel() {
  hidden_size = 0;
}

NonlinearRerankerModel::NonlinearRerankerModel(unsigned hidden_layer_size) {
  hidden_size = hidden_layer_size; 
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
