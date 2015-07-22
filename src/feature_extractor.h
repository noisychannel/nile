#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

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
  unsigned sent_index;
  unsigned hyp_index;
  SimpleDataView* data;
};

class GauravsFeatureExtractor : public KbestFeatureExtractor {
public:
  GauravsFeatureExtractor(GauravDataView* data, Model& cnn_model, const string& source_filename, const string& source_embedding_file, const string& target_embedding_file);
  ~GauravsFeatureExtractor();
  bool MoveToNextSentence();
  bool MoveToNextHypothesis();
  Expression GetFeatures(ComputationGraph& cg) const;
  Expression GetMetricScore(ComputationGraph& cg) const;
  unsigned num_dimensions() const; 
  void Reset();
private:
  GauravDataView* data;
  GauravsModel* gauravs_model;
  unsigned sent_index;
  unsigned hyp_index;
};

class CombinedFeatureExtractor : public KbestFeatureExtractor {
public:
  CombinedFeatureExtractor(const string& kbest_filename, unsigned max_features);
  vector<Expression> ExtractFeatures(const vector<KbestHypothesis>& hyps, ComputationGraph& cg);
  unsigned num_dimensions() const;  
private:
  SimpleKbestFeatureExtractor* simple_extractor;
  GauravsFeatureExtractor* gauravs_extractor;
};
