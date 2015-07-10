#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "utils.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>

#include <iostream>
#include <fstream>
#include <unordered_set>
#include <unordered_map>
#include <climits>
#include <csignal>
#include "pair_sampler.h"

#define NONLINEAR

using namespace std;
using namespace cnn;

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
  if (argc < 3) {
    cerr << "Usage: " << argv[0] << " model kbest.txt" << endl;
    cerr << endl;
    cerr << "Where kbest.txt contains lines of them form" << endl;
    cerr << "sentence_id ||| hypothesis ||| features ||| ... " << endl;
    cerr << "The first three fields must be the sentence id, hypothesis, and features." << endl;
    cerr << "Any extra fields are ignored." << endl;
    cerr << endl;
    cerr << "Here's an example of a valid input line:" << endl;
    cerr << "0 ||| <s> ovatko ne syyt tai ? </s> ||| MaxLexEgivenF=1.26902 Glue=2 LanguageModel=-14.2355 SampleCountF=9.91427 ||| -1.32408" << endl;
    exit(1);
  }
  signal (SIGINT, ctrlc_handler);
  const string model_filename = argv[1];
  const string kbest_filename = argv[2];

  map<string, unsigned> feat2id;

  cerr << "Building model...\n"; 
  cnn::Initialize(argc, argv);
  Model m;
  SimpleSGDTrainer sgd(&m);

  unsigned num_dimensions = 10000; 

  cerr << "Reading model...\n";
  ifstream model_file(model_filename);
  boost::archive::text_iarchive ia(model_file);
  #ifdef NONLINEAR
  unsigned hidden_size;
  ia >> num_dimensions >> hidden_size;
  #else
  ia >> num_dimensions;
  #endif
  vector<float> hyp_features(num_dimensions);

  #ifdef NONLINEAR
  Parameters& p_w1 = *m.add_parameters({hidden_size, num_dimensions});
  Parameters& p_w2 = *m.add_parameters({1, hidden_size});
  Parameters& p_b = *m.add_parameters({hidden_size});

  ComputationGraph hg;
  VariableIndex i_w1 = hg.add_parameters(&p_w1);
  VariableIndex i_w2 = hg.add_parameters(&p_w2);
  VariableIndex i_b = hg.add_parameters(&p_b);
  VariableIndex i_h = hg.add_input({num_dimensions}, &hyp_features); // Hypothesis feature vector
  VariableIndex i_hs1 = hg.add_function<AffineTransform>({i_b, i_w1, i_h}); // Hypothesis score
  VariableIndex i_hs2 = hg.add_function<Tanh>({i_hs1});
  VariableIndex i_hs3 = hg.add_function<Concatenate>({i_hs2});
  VariableIndex i_hs4 = hg.add_function<MatrixMultiply>({i_w2, i_hs3});

  #else
  Parameters& p_w = *m.add_parameters({1, num_dimensions});

  ComputationGraph hg;
  VariableIndex i_w = hg.add_parameters(&p_w); // The weight vector 
  VariableIndex i_h = hg.add_input({num_dimensions}, &hyp_features); // Hypothesis feature vector
  VariableIndex i_hs = hg.add_function<MatrixMultiply>({i_w, i_h}); // Hypothesis score
  #endif

  #ifdef NONLINEAR
  ia >> p_w1 >> p_w2 >> p_b >> feat2id;
  #else
  ia >> p_w >> feat2id;
  #endif


  cerr << "Reranking..." << endl;
  ifstream kbest_file(kbest_filename);
  string best_line = "";
  double best_score = 0.0;
  string current_id = "";

  for (string line; getline(kbest_file, line);) {
    KbestHypothesis hyp = KbestHypothesis::parse(line);
    if (hyp.sentence_id != current_id) {
      if (current_id != "") {
        cout << best_line << endl;
        best_line = "";
        best_score = 0.0;
      }
      cerr << hyp.sentence_id << "\r";
      current_id = hyp.sentence_id;
    }

    map<string, double>& features = hyp.features;
    for (unsigned i = 0; i < num_dimensions; ++i) {
      hyp_features[i] = 0.0;
    }
    for (auto it = features.begin(); it != features.end(); ++it) {
      unsigned feat_id = feat2id[it->first];
      hyp_features[feat_id] = it->second;
    }
    double score = as_scalar(hg.forward());
    if (best_line == "" || score > best_score) {
      best_line = line;
      best_score = score;
    }
  } 
  cout << best_line << endl;
 
  return 0;
}
