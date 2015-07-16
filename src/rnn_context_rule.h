#pragma once
#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "utils.h"
#include "kbest_hypothesis.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <string>

using namespace std;
using namespace cnn;

const unsigned LAYERS = 2;


struct Params {
  LookupParameters* p_w;
  Parameters* p_R;
  Parameters* p_bias;
};


template <class Builder>
struct RNNContextRule {
  // The embeddings for the words
  LookupParameters* p_w_source;
  LookupParameters* p_w_target;
  // The model parameters
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

  explicit RNNContextRule(Model &model, LookupParameters* p_w_s, LookupParameters* p_w_t)
  {
    const unsigned EMBEDDING_DIM = p_w_s->dim[0];
    const unsigned HIDDEN_DIM = EMBEDDING_DIM;
    const unsigned VOCAB_SIZE_SOURCE = p_w_s->values.size();
    const unsigned VOCAB_SIZE_TARGET = p_w_t->values.size();

    builder_context_left = Builder(LAYERS, EMBEDDING_DIM, HIDDEN_DIM, &model);
    builder_context_right = Builder(LAYERS, EMBEDDING_DIM, HIDDEN_DIM, &model);
    builder_rule_source = Builder(LAYERS, EMBEDDING_DIM, HIDDEN_DIM, &model);
    builder_rule_target = Builder(LAYERS, EMBEDDING_DIM, HIDDEN_DIM, &model);
    //TODO: What is the dimnesionality??
    p_w_source = p_w_s;
    p_w_target = p_w_t;
    //p_w_source = model.add_lookup_parameters(VOCAB_SIZE_SOURCE, {EMBEDDING_DIM});
    //p_w_target = model.add_lookup_parameters(VOCAB_SIZE_TARGET, {EMBEDDING_DIM});
    p_R_cl = model.add_parameters({VOCAB_SIZE_SOURCE, HIDDEN_DIM});
    p_bias_cl = model.add_parameters({VOCAB_SIZE_SOURCE});
    p_R_cr = model.add_parameters({VOCAB_SIZE_SOURCE, HIDDEN_DIM});
    p_bias_cr = model.add_parameters({VOCAB_SIZE_SOURCE});
    p_R_rs = model.add_parameters({VOCAB_SIZE_SOURCE, HIDDEN_DIM});
    p_bias_rs = model.add_parameters({VOCAB_SIZE_SOURCE});
    p_R_rt = model.add_parameters({VOCAB_SIZE_TARGET, HIDDEN_DIM});
    p_bias_rt = model.add_parameters({VOCAB_SIZE_TARGET});
  }

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
  Expression BuildRNNGraph(struct Context c, ComputationGraph& hg);

  // Reads a sequence of contexts built for an n-best hypothesis (in association
  // with the source side sentence) and runs the CRNN model to get rule
  // embeddings.
  //
  // The embeddings are currently simply summed together to get the feature
  // vector for the hypothesis. This may change in the future.
  // TODO (gaurav)
  Expression BuildRuleSequenceModel(vector<struct Context> cSeq, ComputationGraph& hg);
};

//vector<int> ReadPhrase(const vector<string> line, Dict* sd);

Expression getRNNRuleContext(
    const vector<unsigned>& src, const vector<unsigned>& tgt,
    const vector<PhraseAlignmentLink>& links,
    LookupParameters* p_w_source, LookupParameters* p_w_target,
    ComputationGraph& hg, Model& model);
