#include <regex>
#include "kbest_hypothesis.h"

KbestHypothesis KbestHypothesis::parse(string input) {
  vector<string> parts = tokenize(input, "|||");
  parts = strip(parts);
  string sentence_id = parts[0];
  string sentence = parts[1];
  map<string, double> features = parse_feature_string(parts[2]);
  double metric_score = std::stod(parts[parts.size() - 1]);
  return {sentence_id, sentence, features, metric_score};
}

vector<string> KbestHypothesis::TokenizedTarget() const {
  vector<PhraseAlignmentLink> phrase_alignment;
  return TokenizedTarget(phrase_alignment);
}

vector<string> KbestHypothesis::TokenizedTarget(vector<PhraseAlignmentLink>& phrase_alignment) const {
  vector<string> tParts = tokenize(sentence, " ");
  vector<string> target_words;
  tParts = strip(tParts);

  smatch sm;
  regex r("\\|(\\d+)-(\\d+)\\|");
  unsigned last_fucker = 0;
  for (auto i = tParts.begin(); i != tParts.end(); ++i) {
    if (regex_match(*i, sm, r)) {
      // Match found : this is alignment information
      assert(sm.size() == 3);
      string tmpFrom = sm[1];
      string tmpTo = sm[2];
      unsigned srcFrom = atoi(tmpFrom.c_str());
      // Add one to convert Moses's inclusive format to exclusive format
      unsigned srcTo = atoi(tmpTo.c_str()) + 1;
      unsigned tgtFrom = last_fucker;
      unsigned tgtTo = target_words.size();
      phrase_alignment.push_back({srcFrom, srcTo, tgtFrom, tgtTo});
      last_fucker = tgtTo;
    }
    else {
      // Target tokens
      // Accumulate these till we see alignment info
      target_words.push_back(*i);
    }
  }

  return target_words;
}
