#include <vector>
#include "gaurav.h"
#include "kbestlist.h"
using namespace std;

class KbestListDataView {
public:
  KbestListDataView();
  explicit KbestListDataView(KbestList* kbest_list);
  virtual ~KbestListDataView();
  virtual unsigned size() const = 0;
};

class SimpleDataView : public KbestListDataView {
public:
  explicit SimpleDataView(KbestList* kbest_list);
  explicit SimpleDataView(KbestList* kbest_list, SimpleDataView* previous);
  explicit SimpleDataView(KbestList* kbest_list, unsigned max_features);
  ~SimpleDataView();
  unsigned size() const;
  unsigned num_hyps(unsigned sent_index);
  Expression GetFeatureVector(unsigned sent_index, unsigned hyp_index, ComputationGraph& cg) const;
  Expression GetSentenceFeatureMatrix(unsigned sent_index, ComputationGraph& cg) const;
  Expression GetMetricScore(unsigned sent_index, unsigned hyp_index, ComputationGraph& cg) const;
  Expression GetSentenceMetricScoreVector(unsigned sent_index, ComputationGraph& cg) const;
  unsigned num_features() const;
private:
  void Initialize(KbestList* kbest_list);
  void ConvertFeatureVector(const KbestHypothesis& hypothesis, vector<float>& out);
  bool AddFeature(const string& feat_name);

  vector<vector<vector<float> > > feature_vectors;
  vector<vector<float> > metric_scores;

  map<string, unsigned> feat2id;
  map<unsigned, string> id2feat;
  unsigned num_features_;
  const unsigned max_features;
};

class GauravDataView : public KbestListDataView {
public:
  explicit GauravDataView(KbestList* kbest_list);
  ~GauravDataView();
  unsigned size() const;
  unsigned num_hyps(unsigned sent_index) const;
  string GetSentenceId(unsigned sent_index) const;
  vector<string> GetTargetString(unsigned sent_index, unsigned hyp_index) const;
  vector<PhraseAlignmentLink> GetAlignment(unsigned sent_index, unsigned hyp_index) const;
  Expression GetMetricScore(unsigned sent_index, unsigned hyp_index, ComputationGraph& cg) const;
private:
  void Initialize(KbestList* kbest_list); 
  vector<string> sentence_ids;
  vector<vector<vector<string> > > target_strings;
  vector<vector<vector<PhraseAlignmentLink> > > alignments;
  vector<vector<float> > metric_scores;
};
