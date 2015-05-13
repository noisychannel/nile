#include "cnn/edges.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "embedding.h"

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
unsigned VOCAB_SIZE = 0;

cnn::Dict d;
cnn::Dict sourceD;
cnn::Dict targetD;

template <class Builder>
struct RNNContextRule {
  // The embeddings for the words
  LookupParameters* p_w;
  // The model parameters
  Parameters* p_R;
  Parameters* p_bias;
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
    p_w = model.add_lookup_parameters(VOCAB_SIZE, {EMBEDDING_DIM});
    p_R = model.add_parameters({VOCAB_SIZE, HIDDEN_DIM});
    p_bias = model.add_parameters({VOCAB_SIZE});
  }

  // Create a graph of the RNN operating over a sequence
  // Return the VariableIndex of the loss
  vector<VariableIndex> BuildRNNGraph(const vector<int>& sent, Hypergraph& hg) {
    const unsigned sentenceLen = sent.size() - 1;
    // Create a new hypergraph
    builder.new_graph(&hg);
    // Start recurrence over the sentence
    builder.start_new_sequence(&hg);
    // Create symbolic nodes to the computational graph
    VariableIndex i_R = hg.add_parameter(p_R);
    VariableIndex i_bias = hg.add_parameter(p_bias);
    vector<VariableIndex> hiddenStates;
    // TODO: Change this when intergrating with the external network
    for (unsigned t = 0; t < sentenceLen; ++t) {
      // Get the embedding for the current input token
      VariableIndex i_x_t = hg.add_lookup(p_w, sent[t]);
      // y_t = RNN(x_t)
      VariableIndex i_y_t = builder.add_input(i_x_t, &hg);
      // r_T = bias + R * y_t
      VariableIndex i_r_t = hg.add_function<Multilinear>({i_bias, i_R, i_y_t});
      hiddenStates.push_back(i_r_t);
    }
    return hiddenStates;
  }
};

// What do I need ? 
//
// 1. RNN source phrase embedding
// 2. RNN context embedding
// 3. Read / Load word embeddings for s and t languages
// 4. Filter these based on the vocab for this task (dict)
// 5. Let each RNN do this in its constructor


int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  //TODO: Need phrase pairs for embedding
  //Read vectors of source, target phrases
  cerr << "Reading training phrases from " << argv[1] << "...\n";
  // Counts the number of lines
  int tlc = 0;
  // Counts the number of tokens
  int ttoks = 0;
  vector< vector<int> > training, dev;

  string line;
  {
    ifstream in(argv[1]);
    assert(in);
    while (getline(in, line)) {
      ++tlc;
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
