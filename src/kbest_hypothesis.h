#pragma once
#include <map>
#include <unordered_map>
#include <string>
#include <vector>
#include "utils.h"

using namespace std;

struct KbestHypothesis {
  string sentence_id;
  string sentence;
  map<string, double> features;
  double metric_score;

  static KbestHypothesis parse(string input);
};

struct FastKbestHypothesis {
  string sentence_id;
  string sentence;
  unordered_map<unsigned, double> features;
  double metric_score;

  static FastKbestHypothesis parse(string input, map<string, unsigned>& feat2id);
};
