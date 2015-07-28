#include <iostream>
#include "feature_extractor.h"
BOOST_CLASS_EXPORT_IMPLEMENT(SimpleKbestFeatureExtractor)
BOOST_CLASS_EXPORT_IMPLEMENT(GauravsFeatureExtractor)
//BOOST_CLASS_EXPORT_IMPLEMENT(CombinedFeatureExtractor)

KbestFeatureExtractor::~KbestFeatureExtractor() {}

SimpleKbestFeatureExtractor::SimpleKbestFeatureExtractor() : data(NULL) {
  Reset();
}

SimpleKbestFeatureExtractor::SimpleKbestFeatureExtractor(SimpleDataView* data) : data(data) {
  Reset();
}

SimpleKbestFeatureExtractor::~SimpleKbestFeatureExtractor() {}

void SimpleKbestFeatureExtractor::InitializeParameters(Model* cnn_model) {}

void SimpleKbestFeatureExtractor::Reset() {
  sent_index = -1;
  hyp_index = -1;
}

void SimpleKbestFeatureExtractor::SetDataPointer(KbestListDataView* data) {
  this->data = dynamic_cast<SimpleDataView*>(data);
  Reset();
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

GauravsFeatureExtractor::GauravsFeatureExtractor() : data(NULL), gauravs_model(NULL) {
  Reset();
}

GauravsFeatureExtractor::GauravsFeatureExtractor(GauravDataView* data, Model& cnn_model, const string& source_embedding_file, const string& target_embedding_file) : data(data), has_parent(false) {
  gauravs_model = new GauravsModel(cnn_model, source_embedding_file, target_embedding_file);
  Reset();
}

GauravsFeatureExtractor::GauravsFeatureExtractor(GauravDataView* data, GauravsFeatureExtractor* parent) : data(data), has_parent(true) {
  gauravs_model = parent->gauravs_model;
  Reset(); 
}

GauravsFeatureExtractor::~GauravsFeatureExtractor() {
  if (has_parent && gauravs_model != NULL) {
    delete gauravs_model;
    gauravs_model = NULL;
  }
}

void GauravsFeatureExtractor::SetDataPointer(KbestListDataView* data) {
  this->data = dynamic_cast<GauravDataView*>(data);
  Reset();
}

void GauravsFeatureExtractor::InitializeParameters(Model* cnn_model) {
  gauravs_model->InitializeParameters(cnn_model);
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
  //Reset cache
  srcExpCache.clear();
  tgtExpCache.clear();
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
  //cerr << "CG : Num nodes = " << cg.nodes.size() << " ||| " << cg.parameter_nodes.size() << endl;
  string sent_id = data->GetSentenceId(sent_index);
  vector<string> src_words = data->GetSourceString(sent_id);
  vector<unsigned> src = gauravs_model->ConvertSourceSentence(src_words);
  vector<string> tgt_words = data->GetTargetString(sent_index, hyp_index);
  vector<unsigned> tgt = gauravs_model->ConvertTargetSentence(tgt_words);
  vector<PhraseAlignmentLink> alignment = data->GetAlignment(sent_index, hyp_index);
  return gauravs_model->GetRuleContext(src, tgt, alignment, cg, srcExpCache, tgtExpCache);
}

Expression GauravsFeatureExtractor::GetMetricScore(ComputationGraph& cg) const {
  return data->GetMetricScore(sent_index, hyp_index, cg);
}

unsigned GauravsFeatureExtractor::num_dimensions() const {
  return gauravs_model->OutputDimension();
}

/*CombinedFeatureExtractor::CombinedFeatureExtractor() : simple_extractor(NULL), gauravs_extractor(NULL) {
  Reset();
}

void CombinedFeatureExtractor::Reset() {
  simple_extractor->Reset();
  gauravs_extractor->Reset();
}*/
