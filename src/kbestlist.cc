#include <algorithm>
#include "kbestlist.h"

KbestList::~KbestList() {}

SimpleKbestList::SimpleKbestList(string filename) : filename(filename) {
  Reset();
}

SimpleKbestList::~SimpleKbestList() {
  Cleanup();
}

void SimpleKbestList::Cleanup() {
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

void SimpleKbestList::Reset() {
  input_file = new ifstream(filename);
  if (input_file == NULL || !input_file->is_open()) {
    cerr << "Unable to open kbest file: " << filename << endl;
    exit(1);
  }
  current_sent_id = "";
  next_hypothesis = NULL;

}

bool SimpleKbestList::NextSet(vector<KbestHypothesis>& out) {
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

KbestListInRam::KbestListInRam(string filename) : simple_kbest(filename), done_loading(false) {
  Reset();
}

KbestListInRam::~KbestListInRam() {}

bool KbestListInRam::NextSet(vector<KbestHypothesis>& out) {
  if (it != hypotheses.end()) {
    assert (it >= hypotheses.begin());
    assert (it < hypotheses.end());
    out = *it;
    it++;
    return true;
  }

  if (!done_loading) {
    vector<KbestHypothesis> temp;
    if (simple_kbest.NextSet(temp)) {
      hypotheses.push_back(temp);
      it = hypotheses.end();
      out = temp;
      return true;
    }
    else {
      done_loading = true;
      return false;
    }
  }

  return false;
}

void KbestListInRam::Reset() {
  it = hypotheses.begin();
}

void KbestListInRam::Shuffle() {
  Reset();
  random_shuffle(hypotheses.begin(), hypotheses.end());
}
