#include <fstream>
#include <iostream>
#include <climits>
#include "gaurav.h"
#include "context.h"
#include "utils.h"

using namespace std;

unsigned GetEmbeddingDimension(string filename) {
  FILE *f;
  long long words, size;
  f = fopen(filename.c_str(), "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    exit(1);
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  assert (size >= 0);
  assert (size <= UINT_MAX);
  return (unsigned)size;
}

unordered_map<unsigned, vector<float>> LoadEmbeddings(string filename, unordered_map<string, unsigned>& dict) {
  cerr << "Loading embeddings from " << filename << " ... " << endl;
  FILE *f;
  float len;
  long long words, size, a, b;
  char ch;
  float *M;
  char *vocab;
  const long long max_w = 50;              // max length of vocabulary entries
  f = fopen(filename.c_str(), "rb");
  if (f == NULL) {
    printf("Embedding Input file not found\n");
    exit(1);
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    exit(1);
  }
  for (b = 0; b < words; b++) {
    fscanf(f, "%s%c", &vocab[b * max_w], &ch);
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
  }

  // Changes made by Gaurav Kumar (gkumar@cs.jhu.edu)
  // Returns a dictionary of word embeddings
  unordered_map<unsigned, vector<float>> embedDict;
  for (int i = 0; i < words; i++) {
    string word = string(&vocab[i * max_w]);
    dict[word] = i;
    vector<float> wordEmbedding;

    for (int b = 0; b< size; b++) {
      wordEmbedding.push_back(M[i * size + b]);
    }
    embedDict[i] = wordEmbedding;
  }

  assert (embedDict.size() == words);
  return embedDict;
}

GauravsModel::GauravsModel() {
  src_embeddings = NULL;
  tgt_embeddings = NULL;
  src_vocab_size = 0;
  tgt_vocab_size = 0;
  p_R_cl = NULL;
  p_bias_cl = NULL;
  p_R_cr = NULL;
  p_bias_cr = NULL;
  p_R_rs = NULL;
  p_bias_rs = NULL;
  p_R_rt = NULL;
  p_bias_rt = NULL;

  p_V = NULL;

  p_R_mlp_1 = NULL;
  p_bias_mlp_1 = NULL;
  p_R_mlp_2 = NULL;
  p_bias_mlp_2 = NULL;

  use_concat_mlp = false;
}

GauravsModel::GauravsModel(Model& cnn_model, const string& src_embedding_filename,
    const string& tgt_embedding_filename, const bool concat_mlp,
    const bool rand_emb) {
  // XXX: We should read these in from somewhere
  hidden_size = 100;
  num_layers = 1;
  src_vocab_size = 50000;
  tgt_vocab_size = 50000;
  src_embedding_dimension = 50;
  tgt_embedding_dimension = 50;

  use_concat_mlp = concat_mlp;
  cerr << "The concat-MLP variation is " << (concat_mlp ? "enabled" : "disabled") << endl;
  cerr << "The use of random embeddings is " << (rand_emb ? "enabled" : "disabled") << endl;

  src_dict.Convert(kUnk);
  src_dict.Convert(kBos);
  src_dict.Convert(kEos);
  tgt_dict.Convert(kUnk);
  tgt_dict.Convert(kBos);
  tgt_dict.Convert(kEos);

  if (! rand_emb) {
    src_embedding_dimension = GetEmbeddingDimension(src_embedding_filename);
    tgt_embedding_dimension = GetEmbeddingDimension(tgt_embedding_filename);
  }
  cerr << "Source embedding dimension is " << src_embedding_dimension << endl;
  cerr << "Target embedding dimension is " << tgt_embedding_dimension << endl;

  InitializeParameters(&cnn_model);

  if (! rand_emb) {
    InitializeEmbeddings(src_embedding_filename, true);
    InitializeEmbeddings(tgt_embedding_filename, false);
  }

}

void GauravsModel::InitializeEmbeddings(const string& filename, bool is_source) {
  Dict* cnn_dict = (is_source ? &src_dict : &tgt_dict);
  LookupParameters* embeddings = (is_source ? src_embeddings : tgt_embeddings);
  unsigned max_vocab_size = (is_source ? src_vocab_size : tgt_vocab_size);
  unsigned embedding_dim = (is_source ? src_embedding_dimension : tgt_embedding_dimension);

  assert (cnn_dict != NULL);
  assert (embeddings != NULL);
  assert (cnn_dict->size() >= 3);
  assert (cnn_dict->Contains(kBos));
  assert (cnn_dict->Contains(kEos));
  assert (cnn_dict->Contains(kUnk));
  assert (max_vocab_size > 0);
  assert (embedding_dim > 0);
  assert (embeddings->values.size() == max_vocab_size);
  assert (embeddings->dim.size() == embedding_dim);

  unordered_map<string, unsigned> word_id_map;
  unordered_map<unsigned, vector<float> > emb_dict = LoadEmbeddings(filename, word_id_map);
  assert (emb_dict.size() > 0);
  assert (emb_dict.begin()->second.size() > 0);

  for (auto& it : word_id_map) {
    if (!cnn_dict->Contains(it.first) && cnn_dict->size() < max_vocab_size) {
      cnn_dict->Convert(it.first);
    }
    if (!cnn_dict->Contains(it.first)) {
      continue;
    }
    unsigned cnn_id = cnn_dict->Convert(it.first);
    assert (cnn_id < max_vocab_size);
    vector<float>& embedding = emb_dict[it.second];
    assert (embedding.size() == embedding_dim);
    embeddings->Initialize(cnn_id, emb_dict[it.second]); 
  }
}

void GauravsModel::InitializeParameters(Model* cnn_model) {
  builder_context_left = FastLSTMBuilder(num_layers, src_embedding_dimension, hidden_size, cnn_model);
  builder_context_right = FastLSTMBuilder(num_layers, src_embedding_dimension, hidden_size, cnn_model);
  builder_rule_source = GRUBuilder(num_layers, src_embedding_dimension, hidden_size, cnn_model);
  builder_rule_target = GRUBuilder(num_layers, tgt_embedding_dimension, hidden_size, cnn_model);

  p_R_cl = cnn_model->add_parameters({hidden_size, hidden_size});
  p_bias_cl = cnn_model->add_parameters({hidden_size});
  p_R_cr = cnn_model->add_parameters({hidden_size, hidden_size});
  p_bias_cr = cnn_model->add_parameters({hidden_size});
  p_R_rs = cnn_model->add_parameters({hidden_size, hidden_size});
  p_bias_rs = cnn_model->add_parameters({hidden_size});
  p_R_rt = cnn_model->add_parameters({hidden_size, hidden_size});
  p_bias_rt = cnn_model->add_parameters({hidden_size});

  p_R_mlp_1 = cnn_model->add_parameters({hidden_size, 4 * hidden_size});
  p_bias_mlp_1 = cnn_model->add_parameters({hidden_size});
  p_R_mlp_2 = cnn_model->add_parameters({hidden_size, hidden_size});
  p_bias_mlp_2 = cnn_model->add_parameters({hidden_size});

  p_V = cnn_model->add_parameters({hidden_size, hidden_size});

  src_embeddings = cnn_model->add_lookup_parameters(src_vocab_size, {src_embedding_dimension});
  tgt_embeddings = cnn_model->add_lookup_parameters(tgt_vocab_size, {tgt_embedding_dimension});
}

void GauravsModel::BuildDictionary(const unordered_map<string, unsigned>& in, Dict& out) {
  unsigned vocab_size = in.size();
  vector<string> vocab_vec(vocab_size);
  for (auto& it : in) {
    unsigned id = it.second;
    string s = it.first;
    vocab_vec[id] = s;
  }
  for (unsigned i = 0; i < vocab_size; ++i) {
    assert (out.size() == i);
    out.Convert(vocab_vec[i]);
  }
  //out.Freeze();
}

Expression GauravsModel::GetRuleContext(const vector<unsigned>& src, const vector<unsigned>& tgt,
    const vector<PhraseAlignmentLink>& alignment, ComputationGraph& cg,
      ExpCache& exp_cache) {
  assert(src.size() > 0);
  vector<unsigned> src2 = src;
  vector<unsigned> tgt2 = tgt;
  vector<PhraseAlignmentLink> alignment2 = alignment;
  src2.insert(src2.begin(), src_dict.Convert("<s>"));
  src2.insert(src2.end(), src_dict.Convert("</s>"));
  tgt2.insert(tgt2.begin(), tgt_dict.Convert("<s>"));
  tgt2.insert(tgt2.end(), tgt_dict.Convert("</s>"));
  for (PhraseAlignmentLink& link : alignment2) {
    link.src_start++;
    link.src_end++;
    link.tgt_start++;
    link.tgt_end++;
  }

  return getRNNRuleContext(src2, tgt2, alignment2, cg, exp_cache);

}

vector<unsigned> GauravsModel::ConvertSourceSentence(const string& sentence) {
  vector<string> words = tokenize(sentence, " ");
  return ConvertSourceSentence(words);
}

vector<unsigned> GauravsModel::ConvertSourceSentence(const vector<string>& words) {
  return ConvertSentence(words, src_dict);
}

vector<unsigned> GauravsModel::ConvertTargetSentence(const string& sentence) {
  vector<string> words = tokenize(sentence, " ");
  return ConvertTargetSentence(words);
}

vector<unsigned> GauravsModel::ConvertTargetSentence(const vector<string>& words) {
  return ConvertSentence(words, tgt_dict);
}

vector<unsigned> GauravsModel::ConvertSentence(const vector<string>& words, Dict& dict) {
  vector<unsigned> r(words.size());
  for (unsigned i = 0; i < words.size(); ++i) {
    if (dict.Contains(words[i])) {
      r[i] = dict.Convert(words[i]);
    }
    else {
      assert (dict.Contains(kUnk));
      r[i] = dict.Convert(kUnk);
    }
  }
  return r;
}

unsigned GauravsModel::OutputDimension() const {
  return hidden_size;
}

Expression GauravsModel::getRNNRuleContext(
    const vector<unsigned>& src, const vector<unsigned>& tgt,
    const vector<PhraseAlignmentLink>& links, ComputationGraph& hg,
    ExpCache& exp_cache) {

  assert (links.size() > 0);
  assert (src.size() > 0);
  assert (tgt.size() > 0);
  vector<Context> contexts = getContext(src, tgt, links);
  return BuildRuleSequenceModel(contexts, hg, exp_cache);
}

Expression GauravsModel::BuildRuleSequenceModel(const vector<Context>& cSeq, ComputationGraph& hg,
    ExpCache& exp_cache) {
  assert (cSeq.size() > 0);
  //TODO; Is this count right ?
  vector<Expression> ruleEmbeddings;
  for (unsigned i = 0; i < cSeq.size(); ++i) {
    const Context& currentContext = cSeq[i];
    Expression currentEmbedding = BuildRNNGraph(currentContext, hg, exp_cache);
    ruleEmbeddings.push_back(currentEmbedding);
  }
  assert (ruleEmbeddings.size() > 0);
  assert (ruleEmbeddings.size() == cSeq.size());
  return sum(ruleEmbeddings);
}

Expression GauravsModel::BuildRNNGraph(Context c, ComputationGraph& hg, ExpCache& exp_cache) {
  vector<Expression> convVector;
  auto srcIt = exp_cache.srcExpCache.find(c.srcIdx);
  //First check to see if the source is cached
  if (srcIt != exp_cache.srcExpCache.end()) {
    convVector = srcIt->second;
  }
  else {

    auto lContextIt = exp_cache.lContextCache.find(get<0>(c.srcIdx));
    if (lContextIt != exp_cache.lContextCache.end()) {
      convVector.push_back(lContextIt->second);
    }
    else {
      //Initialize builders
      builder_context_left.new_graph(hg);
      // Tell the builder that we are about to start a recurrence
      builder_context_left.start_new_sequence();
      // Create the symbolic graph for the unrolled recurrent network
      vector<Expression> hiddens_cl = Recurrence(c.leftContext, hg,
                                  {src_embeddings, p_R_cl, p_bias_cl}, builder_context_left);
      assert (hiddens_cl.size() > 0);
      convVector.push_back(hiddens_cl.back());
      exp_cache.lContextCache.insert(make_pair(get<0>(c.srcIdx), hiddens_cl.back()));
    }

    auto rContextIt = exp_cache.rContextCache.find(get<1>(c.srcIdx));
    if (rContextIt != exp_cache.rContextCache.end()) {
      convVector.push_back(rContextIt->second);
    }
    else {
      builder_context_right.new_graph(hg);
      builder_context_right.start_new_sequence();
      vector<Expression> hiddens_cr = Recurrence(c.rightContext, hg,
                                  {src_embeddings, p_R_cr, p_bias_cr}, builder_context_right);
      assert (hiddens_cr.size() > 0);
      convVector.push_back(hiddens_cr.back());
      exp_cache.rContextCache.insert(make_pair(get<1>(c.srcIdx), hiddens_cr.back()));
    }

    auto sPhraseIt = exp_cache.sPhraseCache.find(c.srcIdx);
    if (sPhraseIt != exp_cache.sPhraseCache.end()) {
      convVector.push_back(sPhraseIt->second);
    }
    else {
      builder_rule_source.new_graph(hg);
      builder_rule_source.start_new_sequence();
      vector<Expression> hiddens_rs = Recurrence(c.sourceRule, hg,
                                  {src_embeddings, p_R_rs, p_bias_rs}, builder_rule_source);
      assert (hiddens_rs.size() > 0);
      convVector.push_back(hiddens_rs.back());
      exp_cache.sPhraseCache.insert(make_pair(c.srcIdx, hiddens_rs.back()));
    }
    exp_cache.srcExpCache.insert(make_pair(c.srcIdx, convVector));
  }

  vector<Expression> currentConv = convVector;

  auto tgtIt = exp_cache.tPhraseCache.find(c.tgtIdx);
  if (tgtIt != exp_cache.tPhraseCache.end()) {
    currentConv.push_back(tgtIt->second);
  }
  else {
    builder_rule_target.new_graph(hg);
    builder_rule_target.start_new_sequence();
    vector<Expression> hiddens_rt = Recurrence(c.targetRule, hg,
                                {tgt_embeddings, p_R_rt, p_bias_rt}, builder_rule_target);
    assert (hiddens_rt.size() > 0);
    currentConv.push_back(hiddens_rt.back());
    exp_cache.tPhraseCache.insert(make_pair(c.tgtIdx, hiddens_rt.back()));
  }

  Expression V = parameter(hg, p_V);

  // Use the concat_MLP variation if specified
  // Default behavior is to sum the vectors and return
  if (use_concat_mlp) {
    Expression mlp_input = concatenate(currentConv);
    Expression mlp_i_R_1 = parameter(hg, p_R_mlp_1);
    Expression mlp_i_bias_1 = parameter(hg, p_bias_mlp_1);
    Expression mlp_i_R_2 = parameter(hg, p_R_mlp_2);
    Expression mlp_i_bias_2 = parameter(hg, p_bias_mlp_2);
    Expression mlp_i_h = rectify(mlp_i_bias_1 + mlp_i_R_1 * mlp_input);
    return V * rectify(mlp_i_bias_2 + mlp_i_R_2 * mlp_i_h);
  }
  return V * sum(currentConv);
}

vector<Expression> GauravsModel::Recurrence(const vector<unsigned>& sequence, ComputationGraph& hg, Params p, RNNBuilder& builder) {
  assert (sequence.size() > 0);
  const unsigned sequenceLen = sequence.size();
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
