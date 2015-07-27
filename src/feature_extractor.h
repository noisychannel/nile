#pragma once
#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/export.hpp>

#include <iostream>

#include "kbestlist.h"
#include "utils.h"
#include "gaurav.h"
#include "dataview.h"

class KbestFeatureExtractor {
public:
  virtual ~KbestFeatureExtractor();
  virtual bool MoveToNextSentence() = 0;
  virtual bool MoveToNextHypothesis() = 0;
  virtual Expression GetFeatures(ComputationGraph& cg) const = 0;
  virtual Expression GetMetricScore(ComputationGraph& cg) const = 0;
  virtual unsigned num_dimensions() const = 0;
  virtual void Reset() = 0;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {}
};

class SimpleKbestFeatureExtractor : public KbestFeatureExtractor {
public:
  SimpleKbestFeatureExtractor(SimpleDataView* data);
  ~SimpleKbestFeatureExtractor();
  bool MoveToNextSentence();
  bool MoveToNextHypothesis();
  void Reset();
  Expression GetFeatures(ComputationGraph& cg) const;
  Expression GetMetricScore(ComputationGraph& cg) const;
  unsigned num_dimensions() const;
private:
  SimpleKbestFeatureExtractor();
  unsigned sent_index;
  unsigned hyp_index;
  SimpleDataView* data;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    boost::serialization::void_cast_register<SimpleKbestFeatureExtractor, KbestFeatureExtractor>();
  } 
};
BOOST_CLASS_EXPORT_KEY(SimpleKbestFeatureExtractor)

class GauravsFeatureExtractor : public KbestFeatureExtractor {
public: 
  GauravsFeatureExtractor(GauravDataView* data, Model& cnn_model, const string& source_embedding_file, const string& target_embedding_file);
  GauravsFeatureExtractor(GauravDataView* data, GauravsFeatureExtractor* parent);
  ~GauravsFeatureExtractor();
  bool MoveToNextSentence();
  bool MoveToNextHypothesis();
  Expression GetFeatures(ComputationGraph& cg) const;
  Expression GetMetricScore(ComputationGraph& cg) const;
  unsigned num_dimensions() const; 
  void Reset();
private:
  GauravsFeatureExtractor();
  GauravDataView* data;
  GauravsModel* gauravs_model;
  mutable map<tuple<unsigned, unsigned>, Expression> srcExpCache;
  mutable map<tuple<unsigned, unsigned>, Expression> tgtExpCache;
  bool has_parent;
  unsigned sent_index;
  unsigned hyp_index;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    boost::serialization::void_cast_register<GauravsFeatureExtractor, KbestFeatureExtractor>();
    ar & gauravs_model;
  } 
};
BOOST_CLASS_EXPORT_KEY(GauravsFeatureExtractor)

/*class CombinedFeatureExtractor : public KbestFeatureExtractor {
public:
  CombinedFeatureExtractor(SimpleDataView* simple_data, GauravDataView* gaurav_data, Model& cnn_model, const string& source_filename, const string& source_embedding_file, const string& target_embedding_file);
  ~CombinedFeatureExtractor();
  bool MoveToNextSentence();
  bool MoveToNextHypothesis();
  Expression GetFeatures(ComputationGraph& cg) const;
  Expression GetMetricScore(ComputationGraph& cg) const;
  unsigned num_dimensions() const; 
  void Reset();
  CombinedFeatureExtractor(const string& kbest_filename, unsigned max_features);
  vector<Expression> ExtractFeatures(const vector<KbestHypothesis>& hyps, ComputationGraph& cg);
private:
  CombinedFeatureExtractor();
  SimpleKbestFeatureExtractor* simple_extractor;
  GauravsFeatureExtractor* gauravs_extractor;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    boost::serialization::void_cast_register<CombinedFeatureExtractor, KbestFeatureExtractor>();
    ar & simple_extractor;
    ar & gauravs_extractor;
  }
};
BOOST_CLASS_EXPORT_KEY(CombinedFeatureExtractor)*/
