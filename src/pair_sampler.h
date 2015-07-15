#include <fstream>
#include <vector>
#include <string>
#include "kbest_hypothesis.h"
#include "kbestlist.h"
using namespace std;

typedef pair<KbestHypothesis*, KbestHypothesis*> HypothesisPair;
class PairSampler {
public:
  PairSampler(string filename, unsigned samples_per_sentence);
  bool next(HypothesisPair& out);
private:
  unsigned samples_per_sentence;
  unsigned samples_taken;
  unsigned failures;
  SimpleKbestList kbest_list;
  vector<KbestHypothesis> current_sent_hypotheses;
};

PairSampler::PairSampler(string filename, unsigned samples_per_sentence) : kbest_list(filename), samples_per_sentence(samples_per_sentence), samples_taken(0) {
  kbest_list.NextSet(current_sent_hypotheses);
}

bool PairSampler::next(HypothesisPair& out) {
  while (samples_taken >= samples_per_sentence || current_sent_hypotheses.size() < 2) {
    if (!kbest_list.NextSet(current_sent_hypotheses)) {
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

    /*if (current_sent_hypotheses[i].metric_score - current_sent_hypotheses[j].metric_score == 0.0) {
      failures++;
      continue;
    }*/

    if (current_sent_hypotheses[i].metric_score > current_sent_hypotheses[j].metric_score) {
      out = make_pair(&current_sent_hypotheses[i], &current_sent_hypotheses[j]);
    }
    else {
      out = make_pair(&current_sent_hypotheses[j], &current_sent_hypotheses[i]);
    }
    assert (out.first->sentence_id == out.second->sentence_id);
    assert (out.first->sentence_id.length() > 0);
    samples_taken++;
    return true;
  }

  samples_taken = samples_per_sentence;
  return next(out);
  //cerr << "Unable to find a hypothesis pair with different metric scores after 100 tries!" << endl;
  //exit(1);
}

typedef pair<FastKbestHypothesis*, FastKbestHypothesis*> FastHypothesisPair;
class FastPairSampler {
private:
  unordered_map<string, vector<FastKbestHypothesis> > kbest_list;
  vector<string> keys;
  vector<string>::iterator current;

  unsigned samples_per_sentence;
  unsigned samples_taken;
  unsigned failures;

  void read_kbest_list(string filename, map<string, unsigned>& feat2id);

public:
  FastPairSampler(string filename, map<string, unsigned>& feat2id, unsigned samples_per_sentence);
  bool next(FastHypothesisPair& out);
  void reset();
};

FastPairSampler::FastPairSampler(string filename, map<string, unsigned>& feat2id, unsigned samples_per_sentence) {
  read_kbest_list(filename, feat2id);
  this->samples_per_sentence = samples_per_sentence;
  reset();
}

void FastPairSampler::reset() {
  samples_taken = 0;
  current = keys.begin();
}

void FastPairSampler::read_kbest_list(string filename, map<string, unsigned>& feat2id) {
  ifstream input_file(filename);
  string line;
  unordered_set<string> key_set;
  while(getline(input_file, line)) {
    FastKbestHypothesis hyp = FastKbestHypothesis::parse(line, feat2id);
    cerr << hyp.sentence_id << "\r";
    kbest_list[hyp.sentence_id].push_back(hyp);
    if (key_set.find(hyp.sentence_id) == key_set.end()) {
      key_set.insert(hyp.sentence_id);
      keys.push_back(hyp.sentence_id);
    }
  }
  input_file.close();
}

bool FastPairSampler::next(FastHypothesisPair& out) {
  if (samples_taken >= samples_per_sentence) {
    current++;
    samples_taken = 0;
  }

  if (current == keys.end()) {
    return false;
  }


  // It's possible that everything in the k-best list has the same score
  // in which case we won't find any pairs with a suitable gap in metric
  // scores. So we try 100 times, and if we don't get anything, we bail.
  vector<FastKbestHypothesis>& current_sent_hypotheses = kbest_list[*current];
  for (unsigned attempts = 0; attempts < 100; attempts++) {
    unsigned i = rand() % current_sent_hypotheses.size();
    unsigned j = rand() % current_sent_hypotheses.size();
 
    /*if (current_sent_hypotheses[i].metric_score - current_sent_hypotheses[j].metric_score == 0.0) {
      failures++;
      continue;
    }*/

    if (current_sent_hypotheses[i].metric_score > current_sent_hypotheses[j].metric_score) {
      out = make_pair(&current_sent_hypotheses[i], &current_sent_hypotheses[j]);
    }
    else {
      out = make_pair(&current_sent_hypotheses[j], &current_sent_hypotheses[i]);
    }
    assert (out.first->sentence_id == out.second->sentence_id);
    assert (out.first->sentence_id.length() > 0);
    samples_taken++;
    return true;
  }
  samples_taken = samples_per_sentence;
  return next(out);
  //cerr << "Unable to find a hypothesis pair with different metric scores after 100 tries!" << endl;
  //exit(1);
}

