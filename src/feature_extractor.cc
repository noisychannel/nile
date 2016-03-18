#include <iostream>
#include "feature_extractor.h"
BOOST_CLASS_EXPORT_IMPLEMENT(SimpleKbestFeatureExtractor)
BOOST_CLASS_EXPORT_IMPLEMENT(ContextSensitiveFeatureExtractor)
BOOST_CLASS_EXPORT_IMPLEMENT(CombinedFeatureExtractor)

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

void SimpleKbestFeatureExtractor::SetDataPointer(KbestListInRamDataView* data) {
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

ContextSensitiveFeatureExtractor::ContextSensitiveFeatureExtractor() : data(NULL), context_sensitive_model(NULL) {
  Reset();
}

ContextSensitiveFeatureExtractor::ContextSensitiveFeatureExtractor(ContextSensitiveDataView* data, Model& cnn_model, const string& source_embedding_file, const string& target_embedding_file, const bool use_concat_mlp, const bool use_rand_emb) : data(data), has_parent(false) {
  context_sensitive_model = new ContextSensitiveModel(cnn_model, source_embedding_file, target_embedding_file, use_concat_mlp, use_rand_emb);
  Reset();
}

ContextSensitiveFeatureExtractor::ContextSensitiveFeatureExtractor(ContextSensitiveDataView* data, ContextSensitiveFeatureExtractor* parent) : data(data), has_parent(true) {
  context_sensitive_model = parent->context_sensitive_model;
  Reset(); 
}

ContextSensitiveFeatureExtractor::~ContextSensitiveFeatureExtractor() {
  if (has_parent && context_sensitive_model != NULL) {
    delete context_sensitive_model;
    context_sensitive_model = NULL;
  }
}

void ContextSensitiveFeatureExtractor::SetDataPointer(KbestListInRamDataView* data) {
  this->data = dynamic_cast<ContextSensitiveDataView*>(data);
  Reset();
}

void ContextSensitiveFeatureExtractor::InitializeParameters(Model* cnn_model) {
  context_sensitive_model->InitializeParameters(cnn_model);
}

void ContextSensitiveFeatureExtractor::Reset() {
  sent_index = -1;
  hyp_index = -1;
}

bool ContextSensitiveFeatureExtractor::MoveToNextSentence() {
  assert (data->size() > 0);
  sent_index++;
  if (sent_index >= data->size()) {
    sent_index = data->size();
    return false;
  }
  hyp_index = -1;
  //Reset cache
  exp_cache.srcExpCache.clear();
  exp_cache.tPhraseCache.clear();
  exp_cache.sPhraseCache.clear();
  exp_cache.lContextCache.clear();
  exp_cache.rContextCache.clear();
  return true;
}

bool ContextSensitiveFeatureExtractor::MoveToNextHypothesis() {
  hyp_index++;
  if (hyp_index >= data->num_hyps(sent_index)) {
    hyp_index = data->num_hyps(sent_index);
    return false;
  }
  return true;
}

Expression ContextSensitiveFeatureExtractor::GetFeatures(ComputationGraph& cg) const {
  //cerr << "CG : Num nodes = " << cg.nodes.size() << " ||| " << cg.parameter_nodes.size() << endl;
  string sent_id = data->GetSentenceId(sent_index);
  vector<string> src_words = data->GetSourceString(sent_id);
  vector<unsigned> src = context_sensitive_model->ConvertSourceSentence(src_words);
  vector<string> tgt_words = data->GetTargetString(sent_index, hyp_index);
  vector<unsigned> tgt = context_sensitive_model->ConvertTargetSentence(tgt_words);
  vector<PhraseAlignmentLink> alignment = data->GetAlignment(sent_index, hyp_index);
  return context_sensitive_model->GetRuleContext(src, tgt, alignment, cg, exp_cache);
}

Expression ContextSensitiveFeatureExtractor::GetMetricScore(ComputationGraph& cg) const {
  return data->GetMetricScore(sent_index, hyp_index, cg);
}

unsigned ContextSensitiveFeatureExtractor::num_dimensions() const {
  return context_sensitive_model->OutputDimension();
}

CombinedFeatureExtractor::CombinedFeatureExtractor() : simple_extractor(NULL), context_sensitive_extractor(NULL) {
}

CombinedFeatureExtractor::CombinedFeatureExtractor(CombinedDataView* data, Model& cnn_model, const string& source_embedding_file, const string& target_embedding_file, const bool use_concat_mlp, const bool use_rand_emb) : data(data) {
  simple_extractor = new SimpleKbestFeatureExtractor(data->simple);
  context_sensitive_extractor = new ContextSensitiveFeatureExtractor(data->context_data, cnn_model, source_embedding_file, target_embedding_file, use_concat_mlp, use_rand_emb);
  Reset();
}

CombinedFeatureExtractor::CombinedFeatureExtractor(CombinedDataView* data, CombinedFeatureExtractor* parent) {
  simple_extractor = new SimpleKbestFeatureExtractor(data->simple);
  context_sensitive_extractor = new ContextSensitiveFeatureExtractor(data->context_data, parent->context_sensitive_extractor);
  Reset();
}

bool CombinedFeatureExtractor::MoveToNextSentence() {
  bool s = simple_extractor->MoveToNextSentence();
  bool g = context_sensitive_extractor->MoveToNextSentence();
  assert (s == g);
  return s;
}

bool CombinedFeatureExtractor::MoveToNextHypothesis() {
  bool s = simple_extractor->MoveToNextHypothesis();
  bool g = context_sensitive_extractor->MoveToNextHypothesis();
  assert (s == g);
  return s;
}

Expression CombinedFeatureExtractor::GetFeatures(ComputationGraph& cg) const {
  Expression simple_feats = simple_extractor->GetFeatures(cg);
  Expression context_feats = context_sensitive_extractor->GetFeatures(cg);
  return concatenate({simple_feats, context_feats});
}

Expression CombinedFeatureExtractor::GetMetricScore(ComputationGraph& cg) const {
  return simple_extractor->GetMetricScore(cg);
}

unsigned CombinedFeatureExtractor::num_dimensions() const {
  return simple_extractor->num_dimensions() + context_sensitive_extractor->num_dimensions();
}

void CombinedFeatureExtractor::Reset() {
  simple_extractor->Reset();
  context_sensitive_extractor->Reset();
}

void CombinedFeatureExtractor::InitializeParameters(Model* cnn_model) {
  simple_extractor->InitializeParameters(cnn_model);
  context_sensitive_extractor->InitializeParameters(cnn_model);
}

void CombinedFeatureExtractor::SetDataPointer(KbestListInRamDataView* data) {
  this->data = dynamic_cast<CombinedDataView*>(data);
  simple_extractor->SetDataPointer(this->data->simple);
  context_sensitive_extractor->SetDataPointer(this->data->context_data);
  Reset();
}

CombinedFeatureExtractor::~CombinedFeatureExtractor() {
  if (context_sensitive_extractor != NULL) {
    delete context_sensitive_extractor;
    context_sensitive_extractor = NULL;
  }
  if (simple_extractor != NULL) {
    delete simple_extractor;
    simple_extractor = NULL;
  }
}
