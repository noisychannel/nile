#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <cassert>
#include "kbest_hypothesis.h"
using namespace std;

class KbestList {
public:
  KbestList(string filename);
  ~KbestList();
  bool NextSet(vector<KbestHypothesis>& out);
private:
  KbestHypothesis* next_hypothesis;
  string current_sent_id;
  ifstream* input_file;
};

