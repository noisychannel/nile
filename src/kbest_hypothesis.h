#include <map>
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

KbestHypothesis KbestHypothesis::parse(string input) {
  vector<string> parts = tokenize(input, "|||");
  parts = strip(parts);
  string sentence_id = parts[0];
  string sentence = parts[1];
  map<string, double> features = parse_feature_string(parts[2]);
  double metric_score = std::stod(parts[parts.size() - 1]);
  return {sentence_id, sentence, features, metric_score};
}

