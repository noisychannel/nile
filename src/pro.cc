/*#include "cnn/edges.h"
#include "cnn/cnn.h"
#include "cnn/training.h"*/
#include "utils.h"

//#include <boost/archive/text_oarchive.hpp>
//#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>
#include <unordered_set>

using namespace std;
//using namespace cnn;

struct KbestHypothesis {
  string sentence_id;
  string sentence;
  map<string, double> features;
  double metric_score;

  static KbestHypothesis parse(string input);
};

KbestHypothesis KbestHypothesis::parse(string input) {
  vector<string> parts = tokenize(input, "|||");
  parts = strip(parts);
  string sentence_id = parts[0];
  string sentence = parts[1];
  map<string, double> features = parse_feature_string(parts[2]);
  double metric_score = std::stod(parts[parts.size() - 1]);
  return {sentence_id, sentence, features, metric_score};
}

unordered_set<string> get_feature_names(string filename) {
  unordered_set<string> feature_names;
  ifstream input_stream(filename);
  for (string line; getline(input_stream, line);) {
    KbestHypothesis hyp = KbestHypothesis::parse(line);
    map<string, double>& features = hyp.features;
    for (map<string, double>::iterator it = features.begin(); it != features.end(); ++it) {
      feature_names.insert(it->first);
    }  
  }
  input_stream.close();
  return feature_names;
}

typedef pair<KbestHypothesis, KbestHypothesis> HypothesisPair;
class PairSampler {
private:
  ifstream* input_file;
  string current_sent_id;
  vector<KbestHypothesis> current_sent_hypotheses;
  KbestHypothesis* next_hypothesis;

  unsigned samples_per_sentence;
  unsigned samples_taken;

public:
  PairSampler(string filename, unsigned samples_per_sentence);
  ~PairSampler();
  bool next(HypothesisPair& out);
private:
  bool read_next_hyp_set();
};

PairSampler::PairSampler(string filename, unsigned samples_per_sentence) {
  input_file = new ifstream(filename);
  current_sent_id = "";
  next_hypothesis = new KbestHypothesis();
  read_next_hyp_set();
  samples_taken = 0;
  this->samples_per_sentence = samples_per_sentence;
}

PairSampler::~PairSampler() {
  if (next_hypothesis != NULL) {
    delete next_hypothesis;
  }
  next_hypothesis = NULL;

  if (input_file != NULL) {
    input_file->close();
    delete input_file;
    input_file = NULL;
  }
}

bool PairSampler::next(HypothesisPair& out) {
  while (samples_taken >= samples_per_sentence || current_sent_hypotheses.size() < 2) {
    if (!read_next_hyp_set()) {
      return false;
    }
    samples_taken = 0;
  }

  // It's possible that everything in the k-best list has the same score
  // in which case we won't find any pairs with a suitable gap in metric
  // scores. So we try 100 times, and if we don't get anything, we bail.
  for (unsigned attempts = 0; attempts < 100; attempts++) {
    unsigned i = rand() % current_sent_hypotheses.size();
    unsigned j = rand() % current_sent_hypotheses.size();
    if (current_sent_hypotheses[i].metric_score - current_sent_hypotheses[j].metric_score) {
      continue;
    }

    out = make_pair(current_sent_hypotheses[i], current_sent_hypotheses[j]);
    assert (out.first.sentence_id == out.second.sentence_id);
    //assert (out.first.sentence_id.length() > 0);
    samples_taken++;
    return true;
  }

  cerr << "Unable to find a hypothesis pair with different metric scores after 100 tries!" << endl;
  exit(1);
}

bool PairSampler::read_next_hyp_set() {
  assert (input_file != NULL && input_file->is_open());
  current_sent_hypotheses.clear();
  if (next_hypothesis == NULL) {
    return false;
  }
  current_sent_hypotheses.push_back(*next_hypothesis);

  string line;
  while(getline(*input_file, line)) {
    KbestHypothesis hyp = KbestHypothesis::parse(line);
    if (current_sent_id == "") {
      current_sent_id = hyp.sentence_id;
    }
    if (hyp.sentence_id != current_sent_id) {
      *next_hypothesis = hyp;
      current_sent_id = hyp.sentence_id;
      return true;
    }
    current_sent_hypotheses.push_back(hyp);
  }

  assert (current_sent_hypotheses.size() != 0);
  if (next_hypothesis != NULL) {
    delete next_hypothesis;
    next_hypothesis = NULL;
  }
  return true;
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
  const string kbest_filename = argv[1];
  unordered_set<string> feature_names = get_feature_names(kbest_filename);
 
  //cnn::Initialize(argc, argv);
  //Model m;
  //SimpleSGDTrainer sgd(&m);

  unsigned num_dimensions = feature_names.size();
  cerr << "Found " << num_dimensions << " features.\n";
  double margin = 1.0;
  /*Parameters& p_w = *m.add_parameters({num_dimensions});

  Hypergraph hg;
  VariableIndex i_w = hg.add_parameter(&p_w);
  vector<float> ref_features(num_dimensions);
  VariableIndex i_r = hg.add_input({num_dimensions}, &ref_features);
  vector<float> hyp_features(num_dimensions);
  VariableIndex i_h = hg.add_input({num_dimensions}, &hyp_features);
  VariableIndex i_rs = hg.add_function<MatrixMultiply>({i_w, i_r});
  VariableIndex i_hs = hg.add_function<MatrixMultiply>({i_w, i_h});
  VariableIndex i_s = hg.add_function<Concatenate>({i_rs, i_hs});
  VariableIndex i_l = hg.add_function<Hinge>({i_s}, 0, margin);*/

  HypothesisPair hyp_pair;
  PairSampler* sampler = new PairSampler(kbest_filename, 10);

  while (sampler->next(hyp_pair)) {
    //cout << hyp_pair.first.sentence_id << " Pair: " << hyp_pair.first.metric_score << ", " << hyp_pair.second.metric_score << endl;
    //cout << hyp_pair.first.metric_score << endl;
  }

  if (sampler != NULL) {
    delete sampler;
  }
  sampler = NULL;

  return 0;
}
