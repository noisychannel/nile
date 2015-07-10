#include "cnn/cnn.h"
#include "cnn/expr.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/export.hpp>

#include <unordered_set>
#include <string>

#include "kbest_hypothesis.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

class RerankerModel {
public:
  void ReadFeatureNames(string kbest_filename);
  virtual void InitalizeParameters(Model& model) = 0;
  virtual void BuildComputationGraph(vector<KbestHypothesis> kbest, ComputationGraph& cg) = 0;
  virtual Expression score(KbestHypothesis hypothesis, ComputationGraph& cg) = 0;

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
private:
  unsigned hidden_size;
  Parameters* p_w1;
  Parameters* p_w2;
  Parameters* p_b;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<RerankerModel>(*this);
  }
};
BOOST_CLASS_EXPORT(LinearRerankerModel);

class NonlinearRerankerModel : public RerankerModel {
public:
  explicit NonlinearRerankerModel(unsigned hidden_layer_size);
private:
  Parameters* p_w;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<RerankerModel>(*this);
  }
};
BOOST_CLASS_EXPORT(NonlinearRerankerModel)

