#include "cnn/edges.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "embedding.h"
#include "kbest_hypothesis.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cnn;

//Hyperparameters
//TODO: Figure out embedding dim from the binary vector file
unsigned LAYERS = 2;
unsigned EMBEDDING_DIM = 50;
unsigned HIDDEN_DIM = EMBEDDING_DIM;
unsigned VOCAB_SIZE_SOURCE = 0;
unsigned VOCAB_SIZE_TARGET = 0;

cnn::Dict d;
cnn::Dict sourceD;
cnn::Dict targetD;

struct Context {
  const vector<int>& leftContext,
  const vector<int>& rightContext,
  const vector<int>& sourceRule,
  const vector<int>& targetRule
};

struct Params {
  LookupParameters* p_w;
  Parameters* p_R;
  Parameters* p_bias;
};

////////// UTIL FUNCTIONS //////////////////
//

////////////////////////////////////////////

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

  explicit RNNContextRule(Model &model) :
    builder_context_left(LAYERS, EMBEDDING_DIM, HIDDEN_DIM, &model),
    builder_context_right(LAYERS, EMBEDDING_DIM, HIDDEN_DIM, &model),
    builder_rule_source(LAYERS, EMBEDDING_DIM, HIDDEN_DIM, &model),
    builder_rule_target(LAYERS, EMBEDDING_DIM, HIDDEN_DIM, &model)
  {
    //TODO: What is the dimnesionality??
    p_w_source = model.add_lookup_parameters(VOCAB_SIZE_SOURCE, {EMBEDDING_DIM});
    p_w_target = model.add_lookup_parameters(VOCAB_SIZE_TARGET, {EMBEDDING_DIM});
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
  vector<VariableIndex> Recurrence(const vector<int>& sequence, Hypergraph& hg, Params p) {
    const unsigned sequenceLen = sequence.size() - 1;
    vector<VariableIndex> hiddenStates;
    VariableIndex i_R = hg.add_parameter(p_R);
    VariableIndex i_bias = hg.add_parameter(p_bias);
    for (unsigned t = 0; t < sequenceLen; ++t) {
      // Get the embedding for the current input token
      VariableIndex i_x_t = hg.add_lookup(p_w, sent[t]);
      // y_t = RNN(x_t)
      VariableIndex i_y_t = builder.add_input(i_x_t, &hg);
      // r_T = bias + R * y_t
      VariableIndex i_r_t = hg.add_function<Multilinear>({i_bias, i_R, i_y_t});
      VariableIndex i_h_t = hg.add_function<Tanh>({i_r_t});
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
  VariableIndex BuildRNNGraph(struct Context c, Hypergraph& hg) {
    //Initialize builders
    builder_context_left.new_graph(&hg);
    builder_context_right.new_graph(&hg);
    builder_rule_source.new_graph(&hg);
    builder_rule_target.new_graph(&hg);
    // Tell the builder that we are about to start a recurrence
    builder_context_left.start_new_sequence(&hg);
    builder_context_right.start_new_sequence(&hg);
    builder_rule_source.start_new_sequence(&hg);
    builder_rule_target.start_new_sequence(&hg);
    // Create the symbolic graph for the unrolled recurrent network
    vector<VariableIndex> hiddens_cl = Recurrence(c.leftContext, hg,
                                {p_w_source, p_R_cl, p_bias_cl});
    vector<VariableIndex> hiddens_cr = Recurrence(c.rightContext, hg,
                                {p_w_source, p_R_cr, p_bias_cr});
    vector<VariableIndex> hiddens_rs = Recurrence(c.sourceRule, hg,
                                {p_w_source, p_R_rs, p_bias_rs});
    vector<VariableIndex> hiddens_rt = Recurrence(c.targetRule, hg,
                                {p_w_target, p_R_rt, p_bias_rt});
    VariableIndex conv = hg.add_function<Sum>({hiddens_cl.back(), hiddens_cr.back(),
                              hiddens_rs.back(), hiddens_rt.back()};
    return conv;
  }

  // Reads a sequence of contexts built for an n-best hypothesis (in association
  // with the source side sentence) and runs the CRNN model to get rule
  // embeddings.
  //
  // The embeddings are currently simply summed together to get the feature
  // vector for the hypothesis. This may change in the future.
  // TODO (gaurav)
  VariableIndex BuildRuleSequenceModel(vector<struct Context> cSeq, Hypergraph& hg) {
    const unsigned cSeqLen = cSeq.size() - 1;
    vector<VariableIndex> ruleEmbeddings;
    for( std::vector<float>::const_iterator i = cSeq.begin(); i != cSeq.end(); ++i) {
      currentContext = *i;
      currentEmbedding = Recurrence(currentContext, hg);
      ruleEmbeddings.push_back(currentEmbedding);
    }
    //TODO: This may be buggy
    return hg.add_function<Sum>(ruleEmbeddings);
  }
};


int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  // Read the source and the n-best file and create a sequence of contexts
  //TODO: Need phrase pairs for embedding
  //Read vectors of source, target phrases
  cerr << "Reading source sentences from " << argv[1] << "...\n";
  cerr << "Read n-best target hyps from " << argv[2] << "...\n";
  // Counts the number of lines
  //int tlc = 0;
  // Counts the number of tokens
  //int ttoks = 0;
  //vector< vector<int> > training, dev;

  string sourceLine;
  string targetHyp;
  string currentSrcID;
  string currentSrc;
    {
      ifstream sourceIn(argv[1]);
      ifstream nBestIn(argv[2]);
      assert(sourceIn);
      assert(nBestIn);
      while (getline(nBestIn, targetHyp)) {
        //++tlc;
        KbestHypothesis hyp = KbestHypothesis::parse(targetHyp);
        if (hyp.sentence_id != currentSrcID) {
          // We have a new n-best sequence
          // First get the new source sentence
          getline((sourceIn, currentSrc));
          currentSrc = ReadLine(currentSrc, &sourceD);
          currentSrcID = hyp.sentence_id;
        }
      }

      // ReadSentence, reads the sentence and creates a dictionary
      training.push_back(ReadSentence(line, &d));
    }
    d.Freeze();
    VOCAB_SIZE = d.size();
    // Get word embeddings for the items in the dictionary
  }

  Model model;
  Trainer* sgd = new SimpleSGDTrainer(&model);
  RNNContextRule<RNNBuilder> rnncr(model);

  unsigned report_every_i = 50;
  unsigned dev_every_i_reports = 500;
  unisgned si = trainining.size();
  vector<unisgned> order(training.size());

}
