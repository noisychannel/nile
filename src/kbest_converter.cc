#include <iostream>
#include <algorithm>
#include <climits>
#include <fstream>
#include <unordered_set>
#include <regex>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "kbest_converter.h"

using namespace std;

// Takes the name of a kbest file name and returns the
// number of unique feature names found therewithin
unordered_set<string> get_feature_names(string filename, unsigned max_size) {
  unordered_set<string> feature_names;
  if (max_size == 0) {
    return feature_names;
  }

  ifstream input_stream(filename);
  for (string line; getline(input_stream, line);) {
    KbestHypothesis hyp = KbestHypothesis::parse(line);
    map<string, double>& features = hyp.features;
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

unordered_set<string> get_feature_names(string filename) {
  return get_feature_names(filename, UINT_MAX);
}

KbestConverter::KbestConverter() {}

KbestConverter::KbestConverter(string kbest_filename) {
  ReadFeatureNames(kbest_filename, UINT_MAX);
}

KbestConverter::KbestConverter(string kbest_filename, unsigned max_features) {
  ReadFeatureNames(kbest_filename, max_features);
}

void KbestConverter::ReadFeatureNames(string kbest_filename, unsigned max_features) {
  cerr << "Reading feature names from k-best list...\n";
  unordered_set<string> feature_names = get_feature_names(kbest_filename, max_features);
  num_dimensions = feature_names.size();
  assert (num_dimensions > 0);
  cerr << "Found " << num_dimensions << " features.\n";

  cerr << "Building feature name-id maps...\n";
  unsigned feat_map_index = feat2id.size();
  for (string name : feature_names) {
    feat2id[name] = feat_map_index;
    id2feat[feat_map_index] = name;
    feat_map_index++;
  }
}

void KbestConverter::ConvertTargetString(KbestHypothesis& hypothesis, vector<string>& target_words) {
  vector<PhraseAlignmentLink> phrase_alignment;
  ConvertTargetString(hypothesis, target_words, phrase_alignment);
}

void KbestConverter::ConvertTargetString(KbestHypothesis& hypothesis, vector<string>& target_words, vector<PhraseAlignmentLink>& phrase_alignment) {
  vector<string> tParts = tokenize(hypothesis.sentence, " ");
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
}

void KbestConverter::ConvertFeatureVector(KbestHypothesis& hypothesis, vector<float>& out) {
  assert (num_dimensions > 0);
  assert (out.size() == num_dimensions);
  fill(out.begin(), out.end(), 0.0);
  for (auto& kvp : hypothesis.features) {
    unsigned feat_id = feat2id[kvp.first];
    assert (feat_id < num_dimensions);
    out[feat_id] = kvp.second;
  }
}

void KbestConverter::ConvertKbestSet(vector<KbestHypothesis>& hyps, vector<vector<float> >& features, vector<float>& scores) {
  assert (num_dimensions > 0);
  features.resize(hyps.size());
  scores.resize(hyps.size());
  for (unsigned i = 0; i < hyps.size(); ++i) {
    features[i].resize(num_dimensions);
    ConvertFeatureVector(hyps[i], features[i]);
    scores[i] = hyps[i].metric_score;
  }
}

