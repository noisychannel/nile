#include <climits>
#include "dataview.h"
BOOST_CLASS_EXPORT_IMPLEMENT(SimpleDataView)
BOOST_CLASS_EXPORT_IMPLEMENT(GauravDataView)
BOOST_CLASS_EXPORT_IMPLEMENT(CombinedDataView)

KbestListDataView::KbestListDataView() {}
KbestListDataView::KbestListDataView(KbestList* kbest_list) {}
KbestListDataView::~KbestListDataView() {}

SimpleDataView::SimpleDataView(KbestList* kbest_list) : max_features(UINT_MAX), num_features_(0) {
  Initialize(kbest_list);
}

SimpleDataView::SimpleDataView(KbestList* kbest_list, SimpleDataView* previous) :  feat2id(previous->feat2id), id2feat(previous->id2feat), num_features_(previous->num_features_), max_features(previous->num_features_) {
  Initialize(kbest_list);
}

SimpleDataView::SimpleDataView(KbestList* kbest_list, unsigned max_features) : max_features(max_features), num_features_(0) {
  Initialize(kbest_list);
}

SimpleDataView::SimpleDataView() : max_features(0), num_features_(0) {}

void SimpleDataView::Initialize(KbestList* kbest_list, const string& source_filename) {
  Initialize(kbest_list);
}

void SimpleDataView::Initialize(KbestList* kbest_list) {
  vector<KbestHypothesis> hypotheses;
  while (kbest_list->NextSet(hypotheses)) {
    vector<vector<float> > feats;
    vector<float> scores;
    for (KbestHypothesis& hyp : hypotheses) {
      vector<float> hyp_feats;
      ConvertFeatureVector(hyp, hyp_feats);
      assert (hyp_feats.size() == num_features_);
      feats.push_back(hyp_feats);
      scores.push_back(hyp.metric_score);
    }
    assert (feats.size() == scores.size());
    feature_vectors.push_back(feats);
    metric_scores.push_back(scores);
  }

  for (auto& sent_feat_vec : feature_vectors) {
    for (auto& hyp_feat_vec : sent_feat_vec) {
      hyp_feat_vec.resize(num_features_);
    }
  }

  assert (feature_vectors[0][0].size() == num_features_);
}

bool SimpleDataView::AddFeature(const string& feat_name) {
  if (num_features_ >= max_features) {
    return false;
  }
  unsigned new_id = feat2id.size();
  feat2id[feat_name] = new_id;
  id2feat[new_id] = feat_name;
  num_features_++;
 
  for (auto& sent_feat_vec : feature_vectors) {
    for (auto& hyp_feat_vec : sent_feat_vec) {
      hyp_feat_vec.resize(num_features_);
    }
  }

  return true;
}

void SimpleDataView::ConvertFeatureVector(const KbestHypothesis& hypothesis, vector<float>& out) {
  fill(out.begin(), out.end(), 0.0);
  for (auto& kvp : hypothesis.features) {
    if (feat2id.find(kvp.first) == feat2id.end()) {
      if (!AddFeature(kvp.first)) {
        continue;
      }
    }

    assert(feat2id.find(kvp.first) != feat2id.end());
    unsigned feat_id = feat2id[kvp.first];
    assert (feat_id < num_features_);
    out.resize(num_features_);
    out[feat_id] = kvp.second;
  }
}

SimpleDataView::~SimpleDataView() {}

unsigned SimpleDataView::size() const {
  assert (feature_vectors.size() == metric_scores.size());
  return feature_vectors.size();
}

unsigned SimpleDataView::num_hyps(unsigned sent_index) {
  assert (sent_index <= feature_vectors.size());
  return feature_vectors[sent_index].size();
}

Expression SimpleDataView::GetFeatureVector(unsigned sent_index, unsigned hyp_index, ComputationGraph& cg) const {
  assert (feature_vectors[sent_index][hyp_index].size() == num_features_);
  return input(cg, {(long)num_features_}, &feature_vectors[sent_index][hyp_index]);
}

Expression SimpleDataView::GetSentenceFeatureMatrix(unsigned sent_index, ComputationGraph& cg) const {
  unsigned n = feature_vectors[sent_index].size();
  vector<Expression> hyp_vectors(n);
  for (unsigned i = 0; i < n; ++i) {
    hyp_vectors[n] = GetFeatureVector(sent_index, n, cg);
  }
  return concatenate_cols(hyp_vectors);
}

Expression SimpleDataView::GetMetricScore(unsigned sent_index, unsigned hyp_index, ComputationGraph& cg) const {
  return input(cg, metric_scores[sent_index][hyp_index]);
}

Expression SimpleDataView::GetSentenceMetricScoreVector(unsigned sent_index, ComputationGraph& cg) const {
  unsigned n = feature_vectors[sent_index].size();
  return input(cg, {(long)n}, &metric_scores[sent_index]);
}

unsigned SimpleDataView::num_features() const {
  return num_features_;
}

GauravDataView::GauravDataView() {}

GauravDataView::GauravDataView(KbestList* kbest_list, const string& source_filename) {
  Initialize(kbest_list, source_filename);
}

GauravDataView::~GauravDataView() {}

void GauravDataView::ReadSource(string filename) {
  ifstream f(filename);
  for (string line; getline(f, line);) {
    vector<string> pieces = tokenize(line, "|||");
    pieces = strip(pieces);
    assert (pieces.size() == 2);
    assert (src_sentences.find(pieces[0]) == src_sentences.end());
    src_sentences[pieces[0]] = tokenize(pieces[1], " ");
  }
  f.close();
}

void GauravDataView::Initialize(KbestList* kbest_list, const string& source_filename) {
  ReadSource(source_filename);
  vector<KbestHypothesis> hypotheses;
  while (kbest_list->NextSet(hypotheses)) {
    assert (hypotheses.size() > 0);
    string sentence_id = hypotheses[0].sentence_id;
    sentence_ids.push_back(sentence_id);
    vector<vector<string> > sent_target_strings;
    vector<vector<PhraseAlignmentLink> > sent_alignments;
    vector<float> sent_metric_scores;
    for (KbestHypothesis& hyp : hypotheses) {
      vector<PhraseAlignmentLink> alignment;
      vector<string> target = hyp.TokenizedTarget(alignment);
      sent_target_strings.push_back(target);
      sent_alignments.push_back(alignment);
      sent_metric_scores.push_back(hyp.metric_score);
    }
    alignments.push_back(sent_alignments);
    target_strings.push_back(sent_target_strings);
    metric_scores.push_back(sent_metric_scores);
    assert (sent_alignments.size() == sent_target_strings.size());
    assert (sent_alignments.size() == sent_metric_scores.size());
  }
  for (const string& id : sentence_ids) {
    assert (src_sentences.find(id) != src_sentences.end());
  }
  assert (alignments.size() == target_strings.size());
  assert (alignments.size() == sentence_ids.size());
  assert (alignments.size() == metric_scores.size());
}

unsigned GauravDataView::size() const {
  return target_strings.size();
}

unsigned GauravDataView::num_hyps(unsigned sent_index) const {
  assert (sent_index < target_strings.size());
  return target_strings[sent_index].size();
}

string GauravDataView::GetSentenceId(unsigned sent_index) const {
  assert (sent_index < target_strings.size());
  return sentence_ids[sent_index];
}

vector<string> GauravDataView::GetSourceString(unsigned sent_index) const {
  return GetSourceString(GetSentenceId(sent_index));
}

vector<string> GauravDataView::GetSourceString(const string& sent_id) const {
  auto it = src_sentences.find(sent_id);
  assert (it != src_sentences.end());
  return it->second;
}

vector<string> GauravDataView::GetTargetString(unsigned sent_index, unsigned hyp_index) const {
  assert (sent_index < target_strings.size());
  assert (hyp_index < target_strings[sent_index].size());
  return target_strings[sent_index][hyp_index];
}

vector<PhraseAlignmentLink> GauravDataView::GetAlignment(unsigned sent_index, unsigned hyp_index) const {
  assert (sent_index < target_strings.size());
  assert (hyp_index < target_strings[sent_index].size());
  return alignments[sent_index][hyp_index];
}

Expression GauravDataView::GetMetricScore(unsigned sent_index, unsigned hyp_index, ComputationGraph& cg) const {
  assert (sent_index < target_strings.size());
  assert (hyp_index < target_strings[sent_index].size());
  return input(cg, metric_scores[sent_index][hyp_index]);
}

CombinedDataView::CombinedDataView(KbestList* kbest_list, const string& source_filename) {
  simple = new SimpleDataView(kbest_list);
  kbest_list->Reset();
  gaurav = new GauravDataView(kbest_list, source_filename);
  assert (simple->size() == gaurav->size());
}

CombinedDataView::~CombinedDataView() {
  if (simple != NULL) {
    delete simple;
    simple = NULL;
  }

  if (gaurav != NULL) {
    delete gaurav;
    gaurav = NULL;
  }
}

unsigned CombinedDataView::size() const {
  unsigned s = simple->size();
  unsigned g = gaurav->size();
  assert (s == g);
  return s;
}

void CombinedDataView::Initialize(KbestList* kbest_list, const string& source_filename) {
  simple->Initialize(kbest_list, source_filename);
  kbest_list->Reset();
  gaurav->Initialize(kbest_list, source_filename);
}

CombinedDataView::CombinedDataView() {
  simple = NULL;
  gaurav = NULL;
}
