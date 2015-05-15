#include "cnn/edges.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/dict.h"
#include "utils.h"
#include "embedding.h"
#include "rnnContextRule.h"

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
#define FAST

using namespace std;
using namespace cnn;

unordered_map<string, vector<int> > read_source(string filename, Dict& src_dict) {
  unordered_map<string, vector<int> > r;
  ifstream f(filename);
  for (string line; getline(f, line);) {
    vector<string> pieces = tokenize(line, "|||");
    assert (pieces.size() == 2);
    assert (r.find(pieces[0]) == r.end());
    r[pieces[0]] = ReadSentence(pieces[1], &src_dict);
  }
  return r;
}

// Takes the name of a kbest file name and returns the
// number of unique feature names found therewithin
unordered_set<string> get_feature_names(string filename, unsigned max_size, Dict& tgt_dict) {
  unordered_set<string> feature_names;
  if (max_size == 0) {
    return feature_names;
  }

  ifstream input_stream(filename);
  for (string line; getline(input_stream, line);) {
    KbestHypothesis hyp = KbestHypothesis::parse(line);
    map<string, double>& features = hyp.features;
    ReadSentence(hyp.sentence, &tgt_dict);
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

unordered_set<string> get_feature_names(string filename, Dict& tgt_dict) {
  return get_feature_names(filename, UINT_MAX, tgt_dict);
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

VariableIndex nonlinear_score_gaurav(Model& m, Hypergraph& hg, vector<int>& source, string& target,
    LookupParameters& w_source, LookupParameters& w_target, VariableIndex& i_w1, VariableIndex& i_w2, VariableIndex& i_b) {
  VariableIndex i_h = getRNNRuleContext(source, target, &w_source, &w_target, hg, m);
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
  const string source_filename = argv[2];
  const string source_embedding_filename = argv[3];
  const string target_embedding_filename = argv[4];
  const unsigned embedding_dimensions = 300;

  cerr << "Reading source sentences...\n";
  Dict src_dict;
  Dict tgt_dict;
  unordered_map<string, vector<int> > source_sentences = read_source(source_filename, src_dict);

  cerr << "Reading feature names from k-best list...\n";
  unordered_set<string> feature_names = get_feature_names(kbest_filename, 1000, tgt_dict);
  unsigned num_dimensions = feature_names.size();
  unsigned samples_per_sentence = 1;
  cerr << "Found " << num_dimensions << " features.\n";

  cerr << "Computing vocab sizes...\n";
  unsigned src_vocab_size = src_dict.size();
  unsigned tgt_vocab_size = tgt_dict.size();
  cerr << "Found " << src_vocab_size << " source types and " << tgt_vocab_size << " target types.\n";

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
  FastPairSampler sampler(kbest_filename, feat2id, samples_per_sentence);

  cerr << "Building model...\n"; 
  cnn::Initialize(argc, argv);
  Model m;
  SimpleSGDTrainer sgd(&m);

  double margin = 1.0;
  double learning_rate = 1.0e-1;
  vector<float> ref_features(num_dimensions);
  vector<float> hyp_features(num_dimensions);

  unsigned hidden_size = 50;
  Parameters& p_w1 = *m.add_parameters({hidden_size, num_dimensions});
  Parameters& p_w2 = *m.add_parameters({1, hidden_size});
  Parameters& p_b = *m.add_parameters({hidden_size});

  map<string, vector<float> > source_embeddings = loadEmbeddings(source_embedding_filename.c_str()); 
  map<string, vector<float> > target_embeddings = loadEmbeddings(target_embedding_filename.c_str());
  LookupParameters& p_s = *m.add_lookup_parameters(src_vocab_size, {embedding_dimensions});
  LookupParameters& p_t = *m.add_lookup_parameters(tgt_vocab_size, {embedding_dimensions});

  for(unsigned i = 0; i < src_dict.size(); ++i) {
    string word = src_dict.Convert(i);
    p_s.Initialize(i, source_embeddings[word]);
  }

  for(unsigned i = 0; i < tgt_dict.size(); ++i) {
    string word = tgt_dict.Convert(i);
    p_s.Initialize(i, target_embeddings[word]);
  }

  cerr << "Training model...\n";
  for (unsigned iteration = 0; iteration < 1; iteration++) {
    sampler.reset();
    FastHypothesisPair hyp_pair; 

    double loss = 0.0;
    while (sampler.next(hyp_pair)) {
      vector<int> src = source_sentences[get<0>(hyp_pair)->sentence_id];
      assert (src.size() > 0);
      vector<int> ref_tgt = ReadSentence(get<0>(hyp_pair)->sentence, &tgt_dict);
      vector<int> hyp_tgt = ReadSentence(get<1>(hyp_pair)->sentence, &tgt_dict);

      cerr << hyp_pair.first->sentence_id << "\r";
      for (unsigned i = 0; i < num_dimensions; ++i) {
        ref_features[i] = 0.0;
        hyp_features[i] = 0.0;
      }

      for (auto it = hyp_pair.first->features.begin(); it != hyp_pair.first->features.end(); ++it) {
        ref_features[it->first] = it->second;
      }
      for (auto it = hyp_pair.second->features.begin(); it != hyp_pair.second->features.end(); ++it) {
        hyp_features[it->first] = it->second;
      }

      Hypergraph hg;
      VariableIndex i_w1 = hg.add_parameter(&p_w1);
      VariableIndex i_w2 = hg.add_parameter(&p_w2);
      VariableIndex i_b = hg.add_parameter(&p_b);
      VariableIndex i_rs = nonlinear_score_gaurav(m, hg, src, get<0>(hyp_pair)->sentence, p_s, p_t, i_w1, i_w2, i_b); // Reference score
      VariableIndex i_hs = nonlinear_score_gaurav(m, hg, src, get<1>(hyp_pair)->sentence, p_s, p_t, i_w1, i_w2, i_b); // Hypothesis score

      VariableIndex i_g = hg.add_function<ConstantMinusX>({i_rs}, margin); // margin - reference_score
      VariableIndex i_l = hg.add_function<Sum>({i_g, i_hs}); // margin - reference_score + hypothesis_score
      VariableIndex i_rl = hg.add_function<Rectify>({i_l}); // max(0, margin - ref_score + hyp_score)

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
  }

/*  boost::archive::text_oarchive oa(cout);
  #ifdef NONLINEAR
  oa << p_w1 << p_w2 << p_b << feat2id;
  #else
  oa << p_w << feat2id;
  #endif */

  return 0;
}
