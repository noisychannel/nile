#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <cassert>
#include "kbest_hypothesis.h"
using namespace std;

class KbestList {
public:
  virtual ~KbestList() = 0;
  virtual bool NextSet(vector<KbestHypothesis>& out) = 0;
  virtual void Reset() = 0;
};

class SimpleKbestListInRam : public KbestList {
public:
  explicit SimpleKbestListInRam(string filename);
  ~SimpleKbestListInRam();
  bool NextSet(vector<KbestHypothesis>& out);
  void Reset();
private:
  void Cleanup();
  KbestHypothesis* next_hypothesis;
  string current_sent_id;
  ifstream* input_file;
  string filename;
};

class KbestListInRam : public KbestList {
public:
  explicit KbestListInRam(string filename);
  ~KbestListInRam();
  bool NextSet(vector<KbestHypothesis>& out);
  void Reset();
  void Shuffle();
  KbestHypothesis Get(unsigned sent_index, unsigned hyp_index) const;
  unsigned BestHypIndex(unsigned sent_index) const;
  int CompareHyps(unsigned sent_index, unsigned hyp_index_a, unsigned hyp_index_b) const;
private:
  bool done_loading;
  SimpleKbestListInRam simple_kbest;
  vector<vector<KbestHypothesis> > hypotheses;
  vector<vector<KbestHypothesis> >::iterator it;
};

