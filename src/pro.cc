#include "cnn/edges.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "utils.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace cnn;

int main(int argc, char** argv) {
  vector<string> parts = tokenize("0 ||| this is |0-2| an example |2-3| . |3-4| ||| lm_score=1.234 tm_score=-1.235 is_crap=1 my_feat=0 test=3.14159 ||| -42", "|||");
  parts = strip(parts);
  for (string token : parts) { 
    cout << "Token: \"" << token << "\"\n";
  }

  string feature_string = parts[2];
  map<string, double> features = parse_feature_string(feature_string);
  for (map<string, double>::iterator it = features.begin(); it != features.end(); ++it) {
    cout << "Feature \"" << it->first << "\" has value " << it->second << "\n";
  }
  return 0;
  cnn::Initialize(argc, argv);
  Model m;
  SimpleSGDTrainer sgd(&m);

  unsigned num_dimensions = 5; // TODO: Read in k-best list and count # of uniq feature IDs
  double margin = 1.0;
  Parameters& p_w = *m.add_parameters({num_dimensions});

  Hypergraph hg;
  VariableIndex i_w = hg.add_parameter(&p_w);
  vector<float> ref_features(num_dimensions);
  VariableIndex i_r = hg.add_input({num_dimensions}, &ref_features);
  vector<float> hyp_features(num_dimensions);
  VariableIndex i_h = hg.add_input({num_dimensions}, &hyp_features);
  VariableIndex i_rs = hg.add_function<MatrixMultiply>({i_w, i_r});
  VariableIndex i_hs = hg.add_function<MatrixMultiply>({i_w, i_h});
  VariableIndex i_s = hg.add_function<Concatenate>({i_rs, i_hs});
  VariableIndex i_l = hg.add_function<Hinge>({i_s}, 0, margin);

  return 0;
}
