#pragma warning
#include "cnn/edges.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "embedding.h"
#include "utils.h"
#include "kbest_hypothesis.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <string>

using namespace std;
using namespace cnn;

//Hyperparameters
unsigned LAYERS = 2;
//TODO: Figure out embedding dim from the binary vector file
unsigned EMBEDDING_DIM = 50;
unsigned HIDDEN_DIM = EMBEDDING_DIM;
unsigned VOCAB_SIZE_SOURCE = 0;
unsigned VOCAB_SIZE_TARGET = 0;

cnn::Dict d;
cnn::Dict sourceD;
cnn::Dict targetD;

struct Context {
  const std::vector<int>& leftContext;
  const std::vector<int>& rightContext;
  const std::vector<int>& sourceRule;
  const std::vector<int>& targetRule;
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

  explicit RNNContextRule(Model &model, LookupParameters* p_w_s, LookupParameters* p_w_t) :
    builder_context_left(LAYERS, EMBEDDING_DIM, HIDDEN_DIM, &model),
    builder_context_right(LAYERS, EMBEDDING_DIM, HIDDEN_DIM, &model),
    builder_rule_source(LAYERS, EMBEDDING_DIM, HIDDEN_DIM, &model),
    builder_rule_target(LAYERS, EMBEDDING_DIM, HIDDEN_DIM, &model)
  {
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
  std::vector<VariableIndex> Recurrence(const std::vector<int>& sequence, Hypergraph& hg, Params p, Builder builder) {
    const unsigned sequenceLen = sequence.size() - 1;
    std::vector<VariableIndex> hiddenStates;
    VariableIndex i_R = hg.add_parameter(p.p_R);
    VariableIndex i_bias = hg.add_parameter(p.p_bias);
    for (unsigned t = 0; t < sequenceLen; ++t) {
      // Get the embedding for the current input token
      VariableIndex i_x_t = hg.add_lookup(p.p_w, sequence[t]);
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
    std::vector<VariableIndex> hiddens_cl = Recurrence(c.leftContext, hg,
                                {p_w_source, p_R_cl, p_bias_cl}, builder_context_left);
    std::vector<VariableIndex> hiddens_cr = Recurrence(c.rightContext, hg,
                                {p_w_source, p_R_cr, p_bias_cr}, builder_context_right);
    std::vector<VariableIndex> hiddens_rs = Recurrence(c.sourceRule, hg,
                                {p_w_source, p_R_rs, p_bias_rs}, builder_rule_source);
    std::vector<VariableIndex> hiddens_rt = Recurrence(c.targetRule, hg,
                                {p_w_target, p_R_rt, p_bias_rt}, builder_rule_target);
    VariableIndex conv = hg.add_function<Sum>({hiddens_cl.back(), hiddens_cr.back(),
                              hiddens_rs.back(), hiddens_rt.back()});
    return conv;
  }

  // Reads a sequence of contexts built for an n-best hypothesis (in association
  // with the source side sentence) and runs the CRNN model to get rule
  // embeddings.
  //
  // The embeddings are currently simply summed together to get the feature
  // vector for the hypothesis. This may change in the future.
  // TODO (gaurav)
  VariableIndex BuildRuleSequenceModel(std::vector<struct Context> cSeq, Hypergraph& hg) {
    //TODO; Is this count right ?
    const unsigned cSeqLen = cSeq.size() - 1;
    std::vector<VariableIndex> ruleEmbeddings;
    for( std::vector<struct Context>::const_iterator i = cSeq.begin(); i != cSeq.end(); ++i) {
      Context currentContext = *i;
      VariableIndex currentEmbedding = BuildRNNGraph(currentContext, hg);
      ruleEmbeddings.push_back(currentEmbedding);
    }
    //TODO: This may be buggy
    return hg.add_function<Sum>(ruleEmbeddings);
  }
};

std::vector<int> ReadPhrase(const std::vector<std::string> line, Dict* sd) {
  std::string word;
  std::vector<int> res;
  for( std::vector<std::string>::const_iterator i = line.begin(); i != line.end(); ++i) {
    word = *i;
    if (word.empty()) break;
    res.push_back(sd->Convert(word));
  }
  return res;
}

std::vector<Context> getContexts(std::string t, vector<int> s) {
  std::vector<Context> contextSeq;
  //std::vector<std::string> sParts = tokenize(s, " ");
  std::vector<int> sParts = s;
  std::vector<std::string> tParts = tokenize(t, " ");
  //sParts = strip(sParts);
  tParts = strip(tParts);
  // Unfortunate use of regex
  std::smatch sm;
  std::regex r("\\|(\\d+)-(\\d+)\\|");
  std::vector<std::string> currentTargetPhrase;
  for( std::vector<std::string>::const_iterator i = tParts.begin(); i != tParts.end(); ++i) {
    if (std::regex_match(*i, sm, r)) {
      // Match found : this is alignment information
      assert(sm.size() == 2);
      std::string tmpFrom = sm[0];
      std::string tmpTo = sm[1];
      unsigned srcFrom = atoi(tmpFrom.c_str());
      unsigned srcTo = atoi(tmpTo.c_str());
      // Get the source phrase
      unsigned srcId = 0;
      std::vector<int> leftContext;
      std::vector<int> rightContext;
      std::vector<int> sourcePhrase;
      for( std::vector<int>::const_iterator i = sParts.begin(); i != sParts.end(); ++i) {
        if (srcId < srcFrom) {
          //Generating left context
          leftContext.push_back(*i);
        }
        else if (srcId >= srcFrom && srcId <= srcTo) {
          // Generating source phrase
          sourcePhrase.push_back(*i);
        }
        else {
          // Generating right context
          rightContext.push_back(*i);
        }
        ++srcId;
      }

      vector<int> tmpTargetPhrase = ReadPhrase(currentTargetPhrase, &targetD);
      Context curContext = {leftContext, rightContext, sourcePhrase, tmpTargetPhrase};
      contextSeq.push_back(curContext);
      // Reset target token collector
      std::vector<std::string> currentTargetPhrase;
    }
    else {
      // Target tokens
      // Accumulate these till we see alignment info
      currentTargetPhrase.push_back(*i);
    }
  }
  return contextSeq;
}


VariableIndex getRNNRuleContext(std::string rawSrc, std::string targetHyp,
  LookupParameters* p_w_source, LookupParameters* p_w_target,
    Hypergraph hg, Model model) {
//int main(int argc, char** argv) {
  //1. Source sentence
  //2. Target Hyp
  //cnn::Initialize(argc, argv);
  // Read the source and the n-best file and create a sequence of contexts
  //std::cerr << "Reading source sentences from " << argv[1] << "...\n";
  //std::cerr << "Read n-best target hyps from " << argv[2] << "...\n";
  // Counts the number of lines
  //int tlc = 0;
  // Counts the number of tokens
  //int ttoks = 0;
  //vector< vector<int> > training, dev;

  std::vector<int> currentSrc = ReadSentence(rawSrc, &sourceD);
  std::vector<Context> contexts = getContexts(targetHyp, currentSrc);
  RNNContextRule<RNNBuilder> rnncr(model, p_w_source, p_w_target);
  return rnncr.BuildRuleSequenceModel(contexts, hg);

  //std::string sourceLine;
  //std::string targetHyp;
  //std::string currentSrcID;
  //std::string rawSrc;
  //vector<int> currentSrc;
  //{
    //ifstream sourceIn(argv[1]);
    //ifstream nBestIn(argv[2]);
    //assert(sourceIn);
    //assert(nBestIn);
    //while (getline(nBestIn, targetHyp)) {
      ////++tlc;
      //KbestHypothesis hyp = KbestHypothesis::parse(targetHyp);
      //if (hyp.sentence_id != currentSrcID) {
        //// We have a new n-best sequence
        //// First get the new source sentence
        //getline(sourceIn, rawSrc);
        //currentSrc = ReadSentence(rawSrc, &sourceD);
        //currentSrcID = hyp.sentence_id;
      //}
      //// targetHyp looks something like this
      //// for |0-0| someone |1-1| to call you |2-3| or something |4-5| . |6-6|
    //}
  //}
  //sourceD.Freeze();
  //targetD.Freeze();
  //VOCAB_SIZE_SOURCE = sourceD.size();
  //VOCAB_SIZE_TARGET = targetD.size();
  //// Read contexts and start recurrence
  //Model model;
  //Trainer* sgd = new SimpleSGDTrainer(&model);
  //Hypergraph hg;
}

int main(int argc, char** argv) {
  std::string rawSrc;
  std::string targetHyp;
  //cnn::Dict sourceD;
  //cnn::Dict targetD;
  LookupParameters* p_w_source;
  LookupParameters* p_w_target;
  Hypergraph hg;
  Model model;
  getRNNRuleContext(rawSrc, targetHyp, p_w_source, p_w_target, hg, model);
}
