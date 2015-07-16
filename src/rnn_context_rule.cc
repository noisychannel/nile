#pragma warning
#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "gaurav.h"
#include "utils.h"
#include "kbest_hypothesis.h"
#include "context.h"
#include "rnn_context_rule.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <string>

using namespace std;
using namespace cnn;

// This is a general recurrence operation for an RNN over a sequence
// Reads in a sequence, creates and returns hidden states.
template <class Builder>
vector<Expression> RNNContextRule<Builder>::Recurrence(const vector<unsigned>& sequence, ComputationGraph& hg, Params p, Builder builder) {
  const unsigned sequenceLen = sequence.size() - 1;
  vector<Expression> hiddenStates;
  Expression i_R = parameter(hg, p.p_R);
  Expression i_bias = parameter(hg, p.p_bias);
  for (unsigned t = 0; t < sequenceLen; ++t) {
    // Get the embedding for the current input token
    Expression i_x_t = lookup(hg, p.p_w, sequence[t]);
    // y_t = RNN(x_t)
    Expression i_y_t = builder.add_input(i_x_t);
    // r_T = bias + R * y_t
    Expression i_r_t = i_bias + i_R * i_y_t;
    Expression i_h_t = tanh(i_r_t);
    hiddenStates.push_back(i_h_t);
  }
  return hiddenStates;
}

// For a given context (source rule, target rule, left context and
// right context, this generates the symbolic graph for the
// operations involving the four RNNs that embed each of these
// into some vector space. Finally, these embeddins are added to
// create the "contextual-rule" embedding.
// This function returns the contextual rule embedding for one context
// instance.
template <class Builder>
Expression RNNContextRule<Builder>::BuildRNNGraph(struct Context c, ComputationGraph& hg) {
  //Initialize builders
  builder_context_left.new_graph(hg);
  builder_context_right.new_graph(hg);
  builder_rule_source.new_graph(hg);
  builder_rule_target.new_graph(hg);
  // Tell the builder that we are about to start a recurrence
  builder_context_left.start_new_sequence();
  builder_context_right.start_new_sequence();
  builder_rule_source.start_new_sequence();
  builder_rule_target.start_new_sequence();
  vector<Expression> convVector;
  // Create the symbolic graph for the unrolled recurrent network
  vector<Expression> hiddens_cl = Recurrence(c.leftContext, hg,
                              {p_w_source, p_R_cl, p_bias_cl}, builder_context_left);
  vector<Expression> hiddens_cr = Recurrence(c.rightContext, hg,
                              {p_w_source, p_R_cr, p_bias_cr}, builder_context_right);
  vector<Expression> hiddens_rs = Recurrence(c.sourceRule, hg,
                              {p_w_source, p_R_rs, p_bias_rs}, builder_rule_source);
  vector<Expression> hiddens_rt = Recurrence(c.targetRule, hg,
                              {p_w_target, p_R_rt, p_bias_rt}, builder_rule_target);
  convVector.push_back(hiddens_cl.back());
  convVector.push_back(hiddens_cr.back());
  convVector.push_back(hiddens_rs.back());
  convVector.push_back(hiddens_rt.back());
  Expression conv = sum(convVector);
  return conv;
}

// Reads a sequence of contexts built for an n-best hypothesis (in association
// with the source side sentence) and runs the CRNN model to get rule
// embeddings.
//
// The embeddings are currently simply summed together to get the feature
// vector for the hypothesis. This may change in the future.
// TODO (gaurav)
template <class Builder>
Expression RNNContextRule<Builder>::BuildRuleSequenceModel(vector<struct Context> cSeq, ComputationGraph& hg) {
  assert (cSeq.size() > 0);
  //TODO; Is this count right ?
  const unsigned cSeqLen = cSeq.size() - 1;
  vector<Expression> ruleEmbeddings;
  for( vector<struct Context>::const_iterator i = cSeq.begin(); i != cSeq.end(); ++i) {
    Context currentContext = *i;
    Expression currentEmbedding = BuildRNNGraph(currentContext, hg);
    ruleEmbeddings.push_back(currentEmbedding);
  }
  //TODO: This may be buggy
  //sum(vector)
  assert (ruleEmbeddings.size() > 0);
  return sum(ruleEmbeddings);
}

Expression getRNNRuleContext(
    const vector<unsigned>& src, const vector<unsigned>& tgt,
    const vector<PhraseAlignmentLink>& links,
    LookupParameters* p_w_source, LookupParameters* p_w_target,
    ComputationGraph& hg, Model& model) {

  vector<Context> contexts = getContext(src, tgt, links);
  RNNContextRule<SimpleRNNBuilder> rnncr(model, p_w_source, p_w_target);
  return rnncr.BuildRuleSequenceModel(contexts, hg);
}
