#include "cnn/edges.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "utils.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>

#include <iostream>
#include <fstream>
#include <unordered_set>
#include <climits>
#include <csignal>
#include "pair_sampler.h"

#define NONLINEAR

using namespace std;
using namespace cnn;

// Takes the name of a kbest file name and returns the
// number of unique feature names found therewithin
unordered_set<string> get_feature_names(string filename, unsigned max_size) {
  unordered_set<string> feature_names;
  if (max_size == 0) {
    return feature_names;
  }

  ifstream input_stream(filename);
  for (string line; getline(input_stream, line);) {
    KbestHypothesis hyp = KbestHypothesis::parse(line);
    map<string, double>& features = hyp.features;
    for (map<string, double>::iterator it = features.begin(); it != features.end(); ++it) {
      feature_names.insert(it->first);
      if (feature_names.size() >= max_size) {
        input_stream.close();
        return feature_names;
      }
    }
    cerr << hyp.sentence_id << "\r";
  }
  input_stream.close();
  return feature_names;
}

unordered_set<string> get_feature_names(string filename) {
  return get_feature_names(filename, UINT_MAX);
}

bool ctrlc_pressed = false;
void ctrlc_handler(int signal) {
  if (ctrlc_pressed) {
    exit(1);
  }
  else {
    ctrlc_pressed = true;
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " kbest.txt" << endl;
    cerr << endl;
    cerr << "Where kbest.txt contains lines of them form" << endl;
    cerr << "sentence_id ||| hypothesis ||| features ||| ... ||| metric score" << endl;
    cerr << "The first three fields must be the sentence id, hypothesis, and features." << endl;
    cerr << "The last field must be the metric score of each hypothesis." << endl;
    cerr << "Any fields in between are ignored." << endl;
    cerr << endl;
    cerr << "Here's an example of a valid input line:" << endl;
    cerr << "0 ||| <s> ovatko ne syyt tai ? </s> ||| MaxLexEgivenF=1.26902 Glue=2 LanguageModel=-14.2355 SampleCountF=9.91427 ||| -1.32408 ||| 21.3" << endl;
    exit(1);
  }
  signal (SIGINT, ctrlc_handler);
  const string kbest_filename = argv[1];

  cerr << "Reading feature names from k-best list...\n";
  unordered_set<string> feature_names = get_feature_names(kbest_filename, 10000);
  unsigned num_dimensions = feature_names.size();
  cerr << "Found " << num_dimensions << " features.\n";

  cerr << "Building feature name-id maps...\n";
  map<string, unsigned> feat2id;
  map<unsigned, string> id2feat;
  unsigned feat_map_index = 0;
  for (string name : feature_names) {
    feat2id[name] = feat_map_index;
    id2feat[feat_map_index] = name;
    feat_map_index++;
  }

  cerr << "Building model...\n"; 
  cnn::Initialize(argc, argv);
  Model m;
  SimpleSGDTrainer sgd(&m);

  double margin = 1.0;
  double learning_rate = 1.0e-1;
  vector<float> ref_features(num_dimensions);
  vector<float> hyp_features(num_dimensions);

  #ifdef NONLINEAR
  unsigned hidden_size = 10;
  Parameters& p_w1 = *m.add_parameters({hidden_size, num_dimensions});
  Parameters& p_w2 = *m.add_parameters({1, hidden_size});
  Parameters& p_b = *m.add_parameters({hidden_size});

  Hypergraph hg;
  VariableIndex i_w1 = hg.add_parameter(&p_w1);
  VariableIndex i_w2 = hg.add_parameter(&p_w2);
  VariableIndex i_b = hg.add_parameter(&p_b);
  VariableIndex i_r = hg.add_input({num_dimensions}, &ref_features); // Reference feature vector
  VariableIndex i_h = hg.add_input({num_dimensions}, &hyp_features); // Hypothesis feature vector
  VariableIndex i_rs1 = hg.add_function<Multilinear>({i_b, i_w1, i_r}); // Reference score
  VariableIndex i_hs1 = hg.add_function<Multilinear>({i_b, i_w1, i_h}); // Hypothesis score
  VariableIndex i_rs2 = hg.add_function<Tanh>({i_rs1});
  VariableIndex i_hs2 = hg.add_function<Tanh>({i_hs1});
  VariableIndex i_rs3 = hg.add_function<Concatenate>({i_rs2});
  VariableIndex i_hs3 = hg.add_function<Concatenate>({i_hs2});
  VariableIndex i_rs4 = hg.add_function<MatrixMultiply>({i_w2, i_rs3});
  VariableIndex i_hs4 = hg.add_function<MatrixMultiply>({i_w2, i_hs3});
  VariableIndex i_g = hg.add_function<ConstantMinusX>({i_rs4}, margin); // margin - reference_score
  VariableIndex i_l = hg.add_function<Sum>({i_g, i_hs4}); // margin - reference_score + hypothesis_score
  VariableIndex i_rl = hg.add_function<Rectify>({i_l}); // max(0, margin - ref_score + hyp_score)

  #else
  Parameters& p_w = *m.add_parameters({1, num_dimensions});

  Hypergraph hg;
  VariableIndex i_w = hg.add_parameter(&p_w); // The weight vector 
  VariableIndex i_r = hg.add_input({num_dimensions}, &ref_features); // Reference feature vector
  VariableIndex i_h = hg.add_input({num_dimensions}, &hyp_features); // Hypothesis feature vector
  VariableIndex i_rs = hg.add_function<MatrixMultiply>({i_w, i_r}); // Reference score
  VariableIndex i_hs = hg.add_function<MatrixMultiply>({i_w, i_h}); // Hypothesis score
  VariableIndex i_g = hg.add_function<ConstantMinusX>({i_rs}, margin); // margin - reference_score
  VariableIndex i_l = hg.add_function<Sum>({i_g, i_hs}); // margin - reference_score + hypothesis_score
  VariableIndex i_rl = hg.add_function<Rectify>({i_l}); // max(0, margin - ref_score + hyp_score)
  #endif

  cerr << "Training model...\n";
  for (unsigned iteration = 0; iteration < 100; iteration++) {
    HypothesisPair hyp_pair;
    PairSampler* sampler = new PairSampler(kbest_filename, 100);

    double loss = 0.0;
    while (sampler->next(hyp_pair)) {
      for (unsigned i = 0; i < num_dimensions; ++i) {
        ref_features[i] = 0.0;
        hyp_features[i] = 0.0;
      }
      for (auto it = hyp_pair.first->features.begin(); it != hyp_pair.first->features.end(); ++it) {
        unsigned id = feat2id[it->first];
        if (id < num_dimensions) { 
          ref_features[id] = it->second;
        }
      }
      for (auto it = hyp_pair.second->features.begin(); it != hyp_pair.second->features.end(); ++it) {
        unsigned id = feat2id[it->first];
        if (id < num_dimensions) {
          hyp_features[id] = it->second;
        }
      }
      loss += as_scalar(hg.forward());
      hg.backward();
      sgd.update(learning_rate);
      if (ctrlc_pressed) {
        break;
      }
    }
    if (ctrlc_pressed) {
      break;
    }
    cerr << "Iteration " << iteration << " loss: " << loss << endl;
    if (sampler != NULL) {
      delete sampler;
    }
    sampler = NULL;
  }

  boost::archive::text_oarchive oa(cout);
  #ifdef NONLINEAR
  oa << p_w1 << p_w2 << p_b << feat2id;
  #else
  oa << p_w << feat2id;
  #endif

  /*cerr << "Final weight vector:" << endl;
  for (unsigned i = 0; i < num_dimensions; ++i) {
    cout << id2feat[i] << " " << TensorTools::AccessElement(p_w.values, {0, i}) << "\n";
  }*/

  return 0;
}
