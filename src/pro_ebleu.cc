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
      if (ctrlc_pressed) {
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

VariableIndex linear_score(ComputationGraph& hg, vector<float>* input, VariableIndex& i_w) {
  unsigned num_dimensions = input->size();
  VariableIndex i_h = hg.add_input({num_dimensions}, input); // Hypothesis feature vector
  VariableIndex i_s = hg.add_function<MatrixMultiply>({i_w, i_h}); // Hypothesis score
  return i_s;
}

VariableIndex nonlinear_score(ComputationGraph& hg, vector<float>* input, VariableIndex& i_w1, VariableIndex& i_w2, VariableIndex& i_b) {
  unsigned num_dimensions = input->size(); 
  VariableIndex i_h = hg.add_input({num_dimensions}, input); // Hypothesis feature vector
  VariableIndex i_g = hg.add_function<AffineTransform>({i_b, i_w1, i_h});
  VariableIndex i_t = hg.add_function<Tanh>({i_g});
  VariableIndex i_s = hg.add_function<MatrixMultiply>({i_w2, i_t});
  return i_s;
}

class KbestList {
public:
  KbestList(string filename) {
    input_file = new ifstream(filename);
    if (input_file == NULL || !input_file->is_open()) {
      cerr << "Unable to open kbest file: " << filename << endl;
      exit(1);
    }
    current_sent_id = "";
    next_hypothesis = NULL;
  }
  ~KbestList() {
    if (input_file != NULL) {
      if (input_file->is_open()) {
        input_file->close();
      }
      delete input_file;
    }
    if (next_hypothesis != NULL) {
      delete next_hypothesis;
    }
  }
  bool NextSet(vector<KbestHypothesis>& out) {
    out.clear();
    if (input_file == NULL) {
      return false;
    }

    if (next_hypothesis != NULL) {
      assert (next_hypothesis->sentence_id.length() > 0);
      out.push_back(*next_hypothesis);
      current_sent_id = next_hypothesis->sentence_id;
      delete next_hypothesis;
      next_hypothesis = NULL;
    }

    string line;
    while(getline(*input_file, line)) {
      KbestHypothesis hyp = KbestHypothesis::parse(line);
      if (current_sent_id == "") {
        current_sent_id = hyp.sentence_id;
      }
      if (hyp.sentence_id != current_sent_id) {
        next_hypothesis = new KbestHypothesis(hyp);
        cerr << current_sent_id << "\r";
        return true;
      }
      out.push_back(hyp);
    }

    assert (out.size() != 0);
    if (input_file != NULL) {
      input_file->close();
      delete input_file;
      input_file = NULL;
    }
    return true;
  }
private:
  KbestHypothesis* next_hypothesis;
  string current_sent_id;
  ifstream* input_file;
};

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " kbest.txt [dev.txt]" << endl;
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
  const string dev_filename = (argc >= 3) ? argv[2] : "";
  cerr << "Running on " << Eigen::nbThreads() << " threads." << endl;

  cerr << "Reading feature names from k-best list...\n";
  unordered_set<string> feature_names = get_feature_names(kbest_filename, 1000);
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
  //SimpleSGDTrainer sgd(&m, 0.0, 0.1);
  AdadeltaTrainer sgd(&m, 0.0, 1e-6);
  sgd.eta_decay = 0.05;

  cerr << "Reading k-best list...\n";

  #ifdef NONLINEAR
  unsigned hidden_size = 50;
  Parameters& p_w1 = *m.add_parameters({hidden_size, num_dimensions});
  Parameters& p_w2 = *m.add_parameters({1, hidden_size});
  Parameters& p_b = *m.add_parameters({hidden_size});
  #else
  Parameters& p_w = *m.add_parameters({1, num_dimensions});
  #endif

 // VariableIndex i_g = hg.add_function<ConstantMinusX>({i_rs}, margin); // margin - reference_score
  //VariableIndex i_l = hg.add_function<Sum>({i_g, i_hs}); // margin - reference_score + hypothesis_score
  //VariableIndex i_rl = hg.add_function<Rectify>({i_l}); // max(0, margin - ref_score + hyp_score)

  cerr << "Training model...\n";
  vector<KbestHypothesis> hypotheses;
  for (unsigned iteration = 0; iteration <= 1000; iteration++) {
    double loss = 0.0;
    unsigned num_sentences = 0;
    KbestList kbest_list(kbest_filename);
    while (kbest_list.NextSet(hypotheses)) {
      num_sentences++;
      ComputationGraph hg;
      #ifdef NONLINEAR
        VariableIndex i_w1 = hg.add_parameters(&p_w1);
        VariableIndex i_w2 = hg.add_parameters(&p_w2);
        VariableIndex i_b = hg.add_parameters(&p_b); 
      #else
        VariableIndex i_w = hg.add_parameters(&p_w); // The weight vector 
      #endif

      assert (hypotheses.size() > 0);
      cerr << hypotheses[0].sentence_id << "\r";
      vector<vector<float> > hypothesis_features(hypotheses.size());
      vector<float> metric_scores(hypotheses.size());
      vector<VariableIndex> hypothesis_scores(hypotheses.size());

      for (unsigned i = 0; i < hypotheses.size(); ++i) {
        assert (hypothesis_features[i].size() == 0);
        hypothesis_features[i].resize(num_dimensions, 0.0f);
        for (auto& kvp : hypotheses[i].features) {
          unsigned feat_id = feat2id[kvp.first];
          hypothesis_features[i][feat_id] = kvp.second;
        }
        metric_scores[i] = hypotheses[i].metric_score;
        #ifdef NONLINEAR
        hypothesis_scores[i] = nonlinear_score(hg, &hypothesis_features[i], i_w1, i_w2, i_b);
        #else
        hypothesis_scores[i] = linear_score(hg, &hypothesis_features[i], i_w);
        #endif
        /*cerr << "Hypothesis feature vector " << i << ": <";
        for (unsigned j = 0; j < hypothesis_features[i].size(); ++j) {
          cerr << ((j == 0) ? "" : " ") << hypothesis_features[i][j];
        }
        cerr << ">" << endl;*/
      }
      VariableIndex score_vector = hg.add_function<Concatenate>(hypothesis_scores);
      VariableIndex hyp_probs = hg.add_function<Softmax>({score_vector});
      vector<float> hps = as_vector(hg.incremental_forward());
      VariableIndex i_refs = hg.add_input({(long)hypotheses.size()}, &metric_scores);
      vector<float> rs = as_vector(hg.incremental_forward());
      /*cerr << "Hyp probs: <";
      for (unsigned i = 0; i < hps.size(); ++i) { cerr << (i == 0 ? "" : " ") << hps[i]; }
      cerr << ">" << endl;
      cerr << "BLEU Scores: <";
      for (unsigned i = 0; i < rs.size(); ++i) { cerr << (i == 0 ? "" : " ") << rs[i]; }
      cerr << ">" << endl;*/
      VariableIndex i_ebleu = hg.add_function<DotProduct>({hyp_probs, i_refs});
      VariableIndex i_loss = hg.add_function<Negate>({i_ebleu});

      loss += as_scalar(hg.incremental_forward());
      if (iteration != 0) {
        hg.backward();
        sgd.update(1.0);
      }
      if (ctrlc_pressed) {
        break;
      }
    }
    if (ctrlc_pressed) {
      break;
    }
    if (dev_filename.length() > 0) {
      //double dev_score = score_devset();
    }
    cerr << "Iteration " << iteration << " loss: " << loss << " (EBLEU = " << -loss / num_sentences << ")" << endl;
  }

  boost::archive::text_oarchive oa(cout);
  #ifdef NONLINEAR
  oa << num_dimensions << hidden_size << p_w1 << p_w2 << p_b << feat2id;
  #else
  oa << num_dimensions << p_w << feat2id;
  #endif

  return 0;
}
