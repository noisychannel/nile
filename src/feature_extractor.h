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
#include "expr_cache.h"
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
  virtual void InitializeParameters(Model* cnn_model) = 0;
  virtual void SetDataPointer(KbestListInRamDataView* data) = 0;

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
  void InitializeParameters(Model* cnn_model);
  void SetDataPointer(KbestListInRamDataView* data);
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
  GauravsFeatureExtractor(GauravDataView* data, Model& cnn_model, const string& source_embedding_file, const string& target_embedding_file, const bool use_concat_mlp, const bool use_rand_emb);
  GauravsFeatureExtractor(GauravDataView* data, GauravsFeatureExtractor* parent);
  ~GauravsFeatureExtractor();
  bool MoveToNextSentence();
  bool MoveToNextHypothesis();
  Expression GetFeatures(ComputationGraph& cg) const;
  Expression GetMetricScore(ComputationGraph& cg) const;
  unsigned num_dimensions() const; 
  void Reset();
  void InitializeParameters(Model* cnn_model);
  void SetDataPointer(KbestListInRamDataView* data);
private:
  GauravsFeatureExtractor();
  GauravDataView* data;
  GauravsModel* gauravs_model;
  bool has_parent;
  unsigned sent_index;
  unsigned hyp_index;
  mutable ExpCache exp_cache;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    boost::serialization::void_cast_register<GauravsFeatureExtractor, KbestFeatureExtractor>();
    ar & gauravs_model;
  } 
};
BOOST_CLASS_EXPORT_KEY(GauravsFeatureExtractor)

class CombinedFeatureExtractor : public KbestFeatureExtractor {
public:
  CombinedFeatureExtractor(CombinedDataView* data, Model& cnn_model, const string& source_embedding_file, const string& target_embedding_file, const bool use_concat_mlp, const bool use_rand_emb);
  CombinedFeatureExtractor(CombinedDataView* data, CombinedFeatureExtractor* parent);
  ~CombinedFeatureExtractor();
  bool MoveToNextSentence();
  bool MoveToNextHypothesis();
  Expression GetFeatures(ComputationGraph& cg) const;
  Expression GetMetricScore(ComputationGraph& cg) const;
  unsigned num_dimensions() const; 
  void Reset();
  void InitializeParameters(Model* cnn_model);
  void SetDataPointer(KbestListInRamDataView* data);
private:
  CombinedFeatureExtractor();
  CombinedDataView* data;
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
BOOST_CLASS_EXPORT_KEY(CombinedFeatureExtractor)
