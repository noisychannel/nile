#pragma once
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <map>
#include <string>
#include <vector>

#include "kbest_hypothesis.h"

struct PhraseAlignmentLink {
  unsigned src_start;
  unsigned src_end;
  unsigned tgt_start;
  unsigned tgt_end;
};

class KbestConverter {
public:
  KbestConverter(const string& kbest_filename);
  KbestConverter(const string& kbest_filename, unsigned max_features);
  static void ConvertTargetString(const KbestHypothesis& hypothesis, vector<string>& target_words);
  static void ConvertTargetString(const KbestHypothesis& hypothesis, vector<string>& target_words, vector<PhraseAlignmentLink>& phrase_alignment);
  void ConvertFeatureVector(const KbestHypothesis& hypothesis, vector<float>& out);
  void ConvertKbestSet(const vector<KbestHypothesis>& hyps, vector<vector<float> >& features, vector<float>& scores);
  unsigned num_dimensions;

private:
  KbestConverter();
  void ReadFeatureNames(const string& kbest_filename, unsigned max_features);
  map<string, unsigned> feat2id;
  map<unsigned, string> id2feat;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) { 
    ar & num_dimensions;
    ar & feat2id;
    ar & id2feat;
  }
};

