#pragma once
#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "cnn/fast-lstm.h"
#include "cnn/gru.h"
#include "context.h"
#include "expr_cache.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

struct Embedding {
  char *vocab;
  float *M;
  long words;
  long size;
};

struct Params {
  LookupParameters* p_w;
  Parameters* p_R;
  Parameters* p_bias;
};

unsigned GetEmbeddingDimension(string filename);
unordered_map<unsigned, vector<float>> LoadEmbeddings(string filename, unordered_map<string, unsigned>& dict);

class ContextSensitiveModel {
public:
  ContextSensitiveModel(Model& cnn_model, const string& src_embedding_filename, const string& tgt_embedding_filename,
      const bool concat_mlp, const bool use_rand_emb);
  void InitializeParameters(Model* cnn_model);
  void InitializeEmbeddings(const string& filename, bool is_source);
  Expression GetRuleContext(const vector<unsigned>& src, const vector<unsigned>& tgt,
      const vector<PhraseAlignmentLink>& alignment, ComputationGraph& cg,
      ExpCache& exp_cache);
  vector<unsigned> ConvertSourceSentence(const string& words);
  vector<unsigned> ConvertSourceSentence(const vector<string>& words);
  vector<unsigned> ConvertTargetSentence(const string& words);
  vector<unsigned> ConvertTargetSentence(const vector<string>& words);
  vector<unsigned> ConvertSentence(const vector<string>& words, Dict& dict);
  unsigned OutputDimension() const;
  const string kUnk = "<unk>";
  const string kBos = "<s>";
  const string kEos = "</s>";
private:
  ContextSensitiveModel();
  bool use_concat_mlp;
  void BuildDictionary(const unordered_map<string, unsigned>& in, Dict& out);
  LookupParameters* src_embeddings;
  LookupParameters* tgt_embeddings;
  unsigned src_vocab_size;
  unsigned tgt_vocab_size;
  Dict src_dict;
  Dict tgt_dict; 
  unsigned src_embedding_dimension;
  unsigned tgt_embedding_dimension;
  unsigned hidden_size;
  unsigned num_layers;

  Parameters* p_R_cl;
  Parameters* p_bias_cl;
  Parameters* p_R_cr;
  Parameters* p_bias_cr;
  Parameters* p_R_rs;
  Parameters* p_bias_rs;
  Parameters* p_R_rt;
  Parameters* p_bias_rt;

  Parameters* p_R_mlp_1;
  Parameters* p_bias_mlp_1;
  Parameters* p_R_mlp_2;
  Parameters* p_bias_mlp_2;

  // Final weight vector for context embeddings
  Parameters* p_V;

  FastLSTMBuilder builder_context_left;
  FastLSTMBuilder builder_context_right;
  GRUBuilder builder_rule_source;
  GRUBuilder builder_rule_target;

  // This is a general recurrence operation for an RNN over a sequence
  // Reads in a sequence, creates and returns hidden states.
  vector<Expression> Recurrence(const vector<unsigned>& sequence, ComputationGraph& hg, Params p, RNNBuilder& builder);

  // For a given context (source rule, target rule, left context and
  // right context, this generates the symbolic graph for the
  // operations involving the four RNNs that embed each of these
  // into some vector space. Finally, these embeddins are added to
  // create the "contextual-rule" embedding.
  // This function returns the contextual rule embedding for one context
  // instance.
  Expression BuildRNNGraph(Context c, ComputationGraph& hg, ExpCache& exp_cache);

  // Reads a sequence of contexts built for an n-best hypothesis (in association
  // with the source side sentence) and runs the CRNN model to get rule
  // embeddings.
  //
  // The embeddings are currently simply summed together to get the feature
  // vector for the hypothesis. This may change in the future.
  Expression BuildRuleSequenceModel(const vector<Context>& cSeq, ComputationGraph& hg,
        ExpCache& exp_cache);

  Expression getRNNRuleContext(
    const vector<unsigned>& src, const vector<unsigned>& tgt,
    const vector<PhraseAlignmentLink>& links, ComputationGraph& hg,
    ExpCache& exp_cache);

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & hidden_size;
    ar & num_layers;
    ar & src_vocab_size;
    ar & tgt_vocab_size;
    ar & src_embedding_dimension;
    ar & tgt_embedding_dimension;
    ar & src_dict;
    ar & tgt_dict;
    ar & use_concat_mlp;
  }
};
