#include "cnn/cnn.h"
#include "cnn/expr.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/map.hpp>

#include <unordered_set>
#include <map>
#include <string>
#include <vector>

#include "kbest_hypothesis.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

class RerankerModel {
public:
  RerankerModel();
  virtual ~RerankerModel();
  void ReadFeatureNames(string kbest_filename);
  void ReadFeatureNames(string kbest_filename, unsigned max_features);
  void ConvertFeatureVector(KbestHypothesis& hypothesis, vector<float>& out);
  void ConvertKbestSet(vector<KbestHypothesis>& hyps, vector<vector<float> >& features, vector<float>& scores);
  void BuildComputationGraph(vector<vector<float> >& features, vector<float>& gold_scores, ComputationGraph& cg);
  virtual void InitializeParameters(Model& model) = 0;
  virtual Expression score(vector<float>* input_features, ComputationGraph& cg) = 0;

protected:
  unordered_set<string> feature_names;
  unsigned num_dimensions;
  map<string, unsigned> feat2id;
  map<unsigned, string> id2feat;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {}
};

class LinearRerankerModel : public RerankerModel {
public:
  void InitializeParameters(Model& model); 
  Expression score(vector<float>* input_features, ComputationGraph& cg);

private:
  Parameters* p_w;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<RerankerModel>(*this);
    ar & num_dimensions;
    ar & p_w;
    ar & feat2id;
  }
};
BOOST_CLASS_EXPORT_KEY(LinearRerankerModel)

class NonlinearRerankerModel : public RerankerModel {
public:
  explicit NonlinearRerankerModel(unsigned hidden_layer_size);
  void InitializeParameters(Model& model);
  Expression score(vector<float>* input_features, ComputationGraph& cg);

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
    ar & num_dimensions;
    ar & hidden_size;
    ar & p_w1;
    ar & p_w2;
    ar & p_b;
    ar & feat2id;
  }
};
BOOST_CLASS_EXPORT_KEY(NonlinearRerankerModel)
