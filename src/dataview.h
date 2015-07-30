#pragma once
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/map.hpp>
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
  virtual void Initialize(KbestList* kbest_list, const string& source_filename) = 0;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {}
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

  void Initialize(KbestList* kbest_list);
  void Initialize(KbestList* kbest_list, const string& source_filename);
private:
  SimpleDataView();
  void ConvertFeatureVector(const KbestHypothesis& hypothesis, vector<float>& out);
  bool AddFeature(const string& feat_name);

  vector<vector<vector<float> > > feature_vectors;
  vector<vector<float> > metric_scores;

  map<string, unsigned> feat2id;
  map<unsigned, string> id2feat;
  unsigned num_features_;
  unsigned max_features;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    boost::serialization::void_cast_register<SimpleDataView, KbestListDataView>();
    max_features = num_features_;
    ar & feat2id;
    ar & id2feat;
    ar & num_features_;
    ar & max_features;
    max_features = num_features_;
  }
};
BOOST_CLASS_EXPORT_KEY(SimpleDataView)

class GauravDataView : public KbestListDataView {
public:
  explicit GauravDataView(KbestList* kbest_list, const string& source_filename);
  ~GauravDataView();
  unsigned size() const;
  unsigned num_hyps(unsigned sent_index) const;
  string GetSentenceId(unsigned sent_index) const;
  vector<string> GetSourceString(unsigned sent_index) const;
  vector<string> GetSourceString(const string& sent_id) const;
  vector<string> GetTargetString(unsigned sent_index, unsigned hyp_index) const;
  vector<PhraseAlignmentLink> GetAlignment(unsigned sent_index, unsigned hyp_index) const;
  Expression GetMetricScore(unsigned sent_index, unsigned hyp_index, ComputationGraph& cg) const;
  void Initialize(KbestList* kbest_list, const string& source_filename); 
private:
  GauravDataView();
  void ReadSource(string filename); 
  vector<string> sentence_ids;
  unordered_map<string, vector<string> > src_sentences;
  vector<vector<vector<string> > > target_strings;
  vector<vector<vector<PhraseAlignmentLink> > > alignments;
  vector<vector<float> > metric_scores;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    boost::serialization::void_cast_register<GauravDataView, KbestListDataView>();
  }
};
BOOST_CLASS_EXPORT_KEY(GauravDataView)

class CombinedDataView : public KbestListDataView {
public:
  CombinedDataView(KbestList* kbest_list, const string& source_filename);
  ~CombinedDataView();
  unsigned size() const;
  void Initialize(KbestList* kbest_list, const string& source_filename);

  SimpleDataView* simple;
  GauravDataView* gaurav;

private:
  CombinedDataView();
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    boost::serialization::void_cast_register<CombinedDataView, KbestListDataView>();
    ar & simple;
    ar & gaurav;
  }
};
BOOST_CLASS_EXPORT_KEY(CombinedDataView)

