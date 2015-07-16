#pragma once
#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include "kbest_converter.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

struct Embedding {
  char *vocab;
  float *M;
  long words;
  long size;
};

unordered_map<unsigned, vector<float>> LoadEmbeddings(string filename, unordered_map<string, unsigned>& dict);

class GauravsModel {
public:
  GauravsModel(string src_filename, string src_embedding_filename, string tgt_embedding_filename);
  void ReadSource(string filename);
  Expression GetRuleContext(const vector<unsigned>& src, const vector<unsigned>& tgt, const vector<PhraseAlignmentLink>& alignment, ComputationGraph& cg, Model& cnn_model);
private:
  unordered_map<string, vector<int> > src_sentences;
  LookupParameters* src_embeddings;
  LookupParameters* tgt_embeddings;
  unsigned src_vocab_size;
  unsigned tgt_vocab_size;
  Dict src_dict;
  Dict tgt_dict;
  const string kUNK = "";
};
