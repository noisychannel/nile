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
#define FAST

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

VariableIndex linear_score(Hypergraph& hg, vector<float>* input, VariableIndex& i_w) {
  unsigned num_dimensions = input->size();
  VariableIndex i_h = hg.add_input({num_dimensions}, input); // Hypothesis feature vector
  VariableIndex i_s = hg.add_function<MatrixMultiply>({i_w, i_h}); // Hypothesis score
  return i_s;
}

VariableIndex nonlinear_score(Hypergraph& hg, vector<float>* input, VariableIndex& i_w1, VariableIndex& i_w2, VariableIndex& i_b) {
  unsigned num_dimensions = input->size(); 
  VariableIndex i_h = hg.add_input({num_dimensions}, input); // Hypothesis feature vector
  VariableIndex i_g = hg.add_function<Multilinear>({i_b, i_w1, i_h});
  VariableIndex i_t = hg.add_function<Tanh>({i_g});
  VariableIndex i_s = hg.add_function<MatrixMultiply>({i_w2, i_t});
  return i_s;
}

int main(int argc, char** argv) {
  srand(0);
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
  unordered_set<string> feature_names = get_feature_names(kbest_filename, 1000);
  unsigned num_dimensions = feature_names.size();
  unsigned samples_per_sentence = 10;
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

  cerr << "Reading k-best list...\n";
  #ifdef FAST
  FastPairSampler sampler(kbest_filename, feat2id, samples_per_sentence);
  #else
  PairSampler* sampler = NULL;
  #endif

  cerr << "Building model...\n"; 
  cnn::Initialize(argc, argv);
  Model m;
  SimpleSGDTrainer sgd(&m);

  double margin = 1.0;
  double learning_rate = 1.0e-1;
  vector<float> ref_features(num_dimensions);
  vector<float> hyp_features(num_dimensions);
  Hypergraph hg;

  #ifdef NONLINEAR
  unsigned hidden_size = 50;
  Parameters& p_w1 = *m.add_parameters({hidden_size, num_dimensions});
  Parameters& p_w2 = *m.add_parameters({1, hidden_size});
  Parameters& p_b = *m.add_parameters({hidden_size});

  VariableIndex i_w1 = hg.add_parameter(&p_w1);
  VariableIndex i_w2 = hg.add_parameter(&p_w2);
  VariableIndex i_b = hg.add_parameter(&p_b);
  VariableIndex i_rs = nonlinear_score(hg, &ref_features, i_w1, i_w2, i_b); // Reference score
  VariableIndex i_hs = nonlinear_score(hg, &hyp_features, i_w1, i_w2, i_b); // Hypothesis score
  #else
  Parameters& p_w = *m.add_parameters({1, num_dimensions});
  VariableIndex i_w = hg.add_parameter(&p_w); // The weight vector 
  VariableIndex i_rs = linear_score(hg, &ref_features, i_w); // Reference score
  VariableIndex i_hs = linear_score(hg, &hyp_features, i_w); // Hypothesis score
  #endif

  VariableIndex i_g = hg.add_function<ConstantMinusX>({i_rs}, margin); // margin - reference_score
  VariableIndex i_l = hg.add_function<Sum>({i_g, i_hs}); // margin - reference_score + hypothesis_score
  VariableIndex i_rl = hg.add_function<Rectify>({i_l}); // max(0, margin - ref_score + hyp_score)

  cerr << "Training model...\n";
  for (unsigned iteration = 0; iteration < 10; iteration++) {
    #ifdef FAST
    sampler.reset();
    FastHypothesisPair hyp_pair; 
    #else
    sampler = new PairSampler(kbest_filename, samples_per_sentence);
    HypothesisPair hyp_pair;
    #endif

    double loss = 0.0;
    #ifdef FAST
    while (sampler.next(hyp_pair)) {
    #else
    while (sampler->next(hyp_pair)) {
    #endif
      cerr << hyp_pair.first->sentence_id << "\r";
      cout << hyp_pair.first->sentence_id << "\n";
      for (unsigned i = 0; i < num_dimensions; ++i) {
        ref_features[i] = 0.0;
        hyp_features[i] = 0.0;
      }
      #ifdef FAST
      for (auto it = hyp_pair.first->features.begin(); it != hyp_pair.first->features.end(); ++it) {
        ref_features[it->first] = it->second;
        //cout << "r" << it->first << " " << it->second << endl;
      }
      for (auto it = hyp_pair.second->features.begin(); it != hyp_pair.second->features.end(); ++it) {
        hyp_features[it->first] = it->second;
        //cout << "h" << it->first << " " << it->second << endl;
      }
      #else
      for (auto it = hyp_pair.first->features.begin(); it != hyp_pair.first->features.end(); ++it) {
        if (feat2id.find(it->first) != feat2id.end()) {
          unsigned id = feat2id[it->first];
          ref_features[id] = it->second;
          //cout << "r" << id << " " << it->second << endl;
        }
      }
      for (auto it = hyp_pair.second->features.begin(); it != hyp_pair.second->features.end(); ++it) {
        if (feat2id.find(it->first) != feat2id.end()) {
          unsigned id = feat2id[it->first];
          hyp_features[id] = it->second;
          //cout << "h" << id << " " << it->second << endl;
        }
      }
      #endif
      loss += as_scalar(hg.forward());
      cout << as_scalar(hg.forward()) << endl;
      hg.backward();
      sgd.update(learning_rate);
      if (ctrlc_pressed) {
        break;
      }
    }
    #ifdef FAST
    #else
    if (sampler != NULL) {
      delete sampler;
      sampler = NULL;
    }
    #endif
    if (ctrlc_pressed) {
      break;
    }
    cerr << "Iteration " << iteration << " loss: " << loss << endl;
    cout << "ITERATION " << iteration << " done." << endl;
  }

/*  boost::archive::text_oarchive oa(cout);
  #ifdef NONLINEAR
  oa << p_w1 << p_w2 << p_b << feat2id;
  #else
  oa << p_w << feat2id;
  #endif */

  return 0;
}
