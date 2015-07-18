#pragma once
#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include "kbest_converter.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "context.h"

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

unordered_map<unsigned, vector<float>> LoadEmbeddings(string filename, unordered_map<string, unsigned>& dict);

template <class Builder>
class GauravsModel {
public:
  GauravsModel(Model& cnn_model, string src_filename, string src_embedding_filename, string tgt_embedding_filename);
  void ReadSource(string filename);
  Expression GetRuleContext(const vector<unsigned>& src, const vector<unsigned>& tgt, const vector<PhraseAlignmentLink>& alignment, ComputationGraph& cg, Model& cnn_model);
  vector<unsigned> GetSourceSentence(const string& sent_id);
  vector<unsigned> ConvertSourceSentence(const string& words);
  vector<unsigned> ConvertSourceSentence(const vector<string>& words);
  vector<unsigned> ConvertTargetSentence(const string& words);
  vector<unsigned> ConvertTargetSentence(const vector<string>& words);
  vector<unsigned> ConvertSentence(const vector<string>& words, Dict& dict);
  unsigned OutputDimension() const;
private:
  void BuildDictionary(const unordered_map<string, unsigned>& in, Dict& out);
  void InitializeParameters(Model& cnn_model);
  unordered_map<string, vector<unsigned> > src_sentences;
  LookupParameters* src_embeddings;
  LookupParameters* tgt_embeddings;
  unsigned src_vocab_size;
  unsigned tgt_vocab_size;
  Dict src_dict;
  Dict tgt_dict;
  const string kUnk = "<unk>";
  unsigned embedding_dimensions;
  const unsigned hidden_size = 71;
  const unsigned LAYERS = 2;

  Parameters* p_R_cl;
  Parameters* p_bias_cl;
  Parameters* p_R_cr;
  Parameters* p_bias_cr;
  Parameters* p_R_rs;
  Parameters* p_bias_rs;
  Parameters* p_R_rt;
  Parameters* p_bias_rt;

  Builder builder_context_left;
  Builder builder_context_right;
  Builder builder_rule_source;
  Builder builder_rule_target;

  // This is a general recurrence operation for an RNN over a sequence
  // Reads in a sequence, creates and returns hidden states.
  vector<Expression> Recurrence(const vector<unsigned>& sequence, ComputationGraph& hg, Params p, Builder builder);

  // For a given context (source rule, target rule, left context and
  // right context, this generates the symbolic graph for the
  // operations involving the four RNNs that embed each of these
  // into some vector space. Finally, these embeddins are added to
  // create the "contextual-rule" embedding.
  // This function returns the contextual rule embedding for one context
  // instance.
  Expression BuildRNNGraph(Context c, ComputationGraph& hg);

  // Reads a sequence of contexts built for an n-best hypothesis (in association
  // with the source side sentence) and runs the CRNN model to get rule
  // embeddings.
  //
  // The embeddings are currently simply summed together to get the feature
  // vector for the hypothesis. This may change in the future.
  // TODO (gaurav)
  Expression BuildRuleSequenceModel(const vector<Context>& cSeq, ComputationGraph& hg);

  Expression getRNNRuleContext(
    const vector<unsigned>& src, const vector<unsigned>& tgt,
    const vector<PhraseAlignmentLink>& links, ComputationGraph& hg);
};
