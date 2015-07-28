#pragma once
#include <map>
#include <unordered_map>
#include <string>
#include <vector>
#include "utils.h"

using namespace std;

struct PhraseAlignmentLink {
  unsigned src_start;
  unsigned src_end;
  unsigned tgt_start;
  unsigned tgt_end;
};

struct KbestHypothesis {
  string sentence_id;
  string sentence;
  map<string, double> features;
  double metric_score;

  vector<string> TokenizedTarget() const;
  vector<string> TokenizedTarget(vector<PhraseAlignmentLink>& phrase_alignment) const;

  static KbestHypothesis parse(string input);
};
