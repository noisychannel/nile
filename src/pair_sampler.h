#include <fstream>
#include <vector>
#include <string>
#include "kbest_hypothesis.h"
using namespace std;

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
  next_hypothesis = NULL;
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

    if (current_sent_hypotheses[i].metric_score - current_sent_hypotheses[j].metric_score == 0.0) {
      continue;
    }

    if (current_sent_hypotheses[i].metric_score > current_sent_hypotheses[j].metric_score) {
      out = make_pair(current_sent_hypotheses[i], current_sent_hypotheses[j]);
    }
    else {
      out = make_pair(current_sent_hypotheses[j], current_sent_hypotheses[i]);
    }
    assert (out.first.sentence_id == out.second.sentence_id);
    //assert (out.first.sentence_id.length() > 0);
    samples_taken++;
    return true;
  }

  cerr << "Unable to find a hypothesis pair with different metric scores after 100 tries!" << endl;
  exit(1);
}

bool PairSampler::read_next_hyp_set() { 
  if (input_file == NULL) {
    return false;
  }
  assert (input_file->is_open());
  current_sent_hypotheses.clear();

  if (next_hypothesis != NULL) {
    assert (next_hypothesis->sentence_id.length() > 0);
    current_sent_hypotheses.push_back(*next_hypothesis);
  }

  string line;
  while(getline(*input_file, line)) {
    KbestHypothesis hyp = KbestHypothesis::parse(line);
    if (current_sent_id == "") {
      current_sent_id = hyp.sentence_id;
    }
    if (hyp.sentence_id != current_sent_id) {
      next_hypothesis = new KbestHypothesis(hyp);
      current_sent_id = hyp.sentence_id;
      cerr << current_sent_id << "\r";
      return true;
    }
    current_sent_hypotheses.push_back(hyp);
  }
 
  assert (current_sent_hypotheses.size() != 0);
  if (input_file != NULL) {
    input_file->close();
    delete input_file;
    input_file = NULL;
  }

  if (next_hypothesis != NULL) {
    delete next_hypothesis;
    next_hypothesis = NULL;
  } 
  return true;
}

