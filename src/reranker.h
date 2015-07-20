#pragma once
#include "cnn/cnn.h"
#include "cnn/expr.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/export.hpp>

#include <unordered_set>
#include <vector>

using namespace std;
using namespace cnn;
using namespace cnn::expr;


class RerankerModel {
public:
  explicit RerankerModel(unsigned num_dimensions);
  Expression BatchScore(vector<vector<float> >& features, ComputationGraph& cg);
  Expression BatchScore(vector<Expression>& features, ComputationGraph& cg);
  void BuildComputationGraph(vector<vector<float> >& features, vector<float>& gold_scores, ComputationGraph& cg);
  void BuildComputationGraph(vector<Expression>& features, vector<float>& gold_scores, ComputationGraph& cg);
  void BuildComputationGraph(vector<Expression>& features, Expression& gold_scores, ComputationGraph& cg);

  virtual ~RerankerModel();
  Expression score(vector<float>* input_features, ComputationGraph& cg);
  virtual Expression score(Expression h, ComputationGraph& cg) = 0;

  Model cnn_model;

protected:
  RerankerModel();
  virtual void InitializeParameters() = 0;
  unsigned num_dimensions;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & num_dimensions;
  }
};

class LinearRerankerModel : public RerankerModel {
public:
  explicit LinearRerankerModel(unsigned num_dimensions);
  void InitializeParameters();
  Expression score(Expression h, ComputationGraph& cg);

private:
  LinearRerankerModel();
  Parameters* p_w;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<RerankerModel>(*this);
    ar & p_w;
  }
};
BOOST_CLASS_EXPORT_KEY(LinearRerankerModel)

class NonlinearRerankerModel : public RerankerModel {
public:
  explicit NonlinearRerankerModel(unsigned num_dimensions, unsigned hidden_layer_size);
  void InitializeParameters();
  Expression score(Expression h, ComputationGraph& cg);

private:
  NonlinearRerankerModel(); 
  unsigned hidden_size;
  Parameters* p_w1;
  Parameters* p_w2;
  Parameters* p_b;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<RerankerModel>(*this);
    ar & hidden_size;
    ar & p_w1;
    ar & p_w2;
    ar & p_b;
  }
};
BOOST_CLASS_EXPORT_KEY(NonlinearRerankerModel)
