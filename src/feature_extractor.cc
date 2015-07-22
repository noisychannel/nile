#include <iostream>
#include "feature_extractor.h"

KbestFeatureExtractor::~KbestFeatureExtractor() {}

SimpleKbestFeatureExtractor::SimpleKbestFeatureExtractor(SimpleDataView* data) : data(data) {
  Reset();
}

SimpleKbestFeatureExtractor::~SimpleKbestFeatureExtractor() {}

void SimpleKbestFeatureExtractor::Reset() {
  sent_index = -1;
  hyp_index = -1;
}

bool SimpleKbestFeatureExtractor::MoveToNextSentence() {
  assert (data->size() > 0);
  sent_index++;
  if (sent_index >= data->size()) {
    sent_index = data->size();
    return false;
  }
  hyp_index = -1;
  return true;
}

bool SimpleKbestFeatureExtractor::MoveToNextHypothesis() {
  hyp_index++;
  if (hyp_index >= data->num_hyps(sent_index)) {
    hyp_index = data->num_hyps(sent_index);
    return false;
  }
  return true;
}

Expression SimpleKbestFeatureExtractor::GetFeatures(ComputationGraph& cg) const {
  assert (sent_index >= 0 && sent_index < data->size());
  assert (hyp_index >= 0 && hyp_index < data->num_hyps(sent_index));
  return data->GetFeatureVector(sent_index, hyp_index, cg);
}

Expression SimpleKbestFeatureExtractor::GetMetricScore(ComputationGraph& cg) const {
  assert (sent_index >= 0 && sent_index < data->size());
  assert (hyp_index >= 0 && hyp_index < data->num_hyps(sent_index));
  return data->GetMetricScore(sent_index, hyp_index, cg);
}

unsigned SimpleKbestFeatureExtractor::num_dimensions() const {
  return data->num_features();
}

GauravsFeatureExtractor::GauravsFeatureExtractor(GauravDataView* data, Model& cnn_model, const string& source_filename, const string& source_embedding_file, const string& target_embedding_file) : data(data) {
  gauravs_model = new GauravsModel(cnn_model, source_filename, source_embedding_file, target_embedding_file);
  Reset();
}

GauravsFeatureExtractor::~GauravsFeatureExtractor() {
  if (gauravs_model != NULL) {
    delete gauravs_model;
    gauravs_model = NULL;
  }
}

void GauravsFeatureExtractor::Reset() {
  sent_index = -1;
  hyp_index = -1;
}

bool GauravsFeatureExtractor::MoveToNextSentence() {
  assert (data->size() > 0);
  sent_index++;
  if (sent_index >= data->size()) {
    sent_index = data->size();
    return false;
  }
  hyp_index = -1;
  return true;
}

bool GauravsFeatureExtractor::MoveToNextHypothesis() {
  hyp_index++;
  if (hyp_index >= data->num_hyps(sent_index)) {
    hyp_index = data->num_hyps(sent_index);
    return false;
  }
  return true;
}

Expression GauravsFeatureExtractor::GetFeatures(ComputationGraph& cg) const {
  string sent_id = data->GetSentenceId(sent_index);
  vector<unsigned> src = gauravs_model->GetSourceSentence(sent_id);
  vector<string> tgt_words = data->GetTargetString(sent_index, hyp_index);
  vector<unsigned> tgt = gauravs_model->ConvertTargetSentence(tgt_words);
  vector<PhraseAlignmentLink> alignment = data->GetAlignment(sent_index, hyp_index);
  return gauravs_model->GetRuleContext(src, tgt, alignment, cg);
}

Expression GauravsFeatureExtractor::GetMetricScore(ComputationGraph& cg) const {
  return data->GetMetricScore(sent_index, hyp_index, cg);
}

unsigned GauravsFeatureExtractor::num_dimensions() const {
  return gauravs_model->OutputDimension();
}