#include <fstream>
#include <iostream>
#include "gaurav.h"
#include "utils.h"
#include "rnn_context_rule.h"

using namespace std;

unordered_map<unsigned, vector<float>> LoadEmbeddings(string filename, unordered_map<string, unsigned>& dict) {
  cerr << "Loading embeddings from " << filename << " ... ";
  FILE *f;
  float len;
  long long words, size, a, b;
  char ch;
  float *M;
  char *vocab;
  const long long max_w = 50;              // max length of vocabulary entries
  f = fopen(filename.c_str(), "rb");
  if (f == NULL) {
    printf("Input file not found\n");
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

GauravsModel::GauravsModel(string src_filename, string src_embedding_filename, string tgt_embedding_filename) {
  unordered_map<string, unsigned> tmp_src_dict, tmp_tgt_dict;
  src_embeddings = LoadEmbeddings(src_embedding_filename, tmp_src_dict);
  tgt_embeddings = LoadEmbeddings(tgt_embedding_filename, tmp_tgt_dict);

  ReadSource(src_filename);

  src_vocab_size = src_embeddings.size();
  tgt_vocab_size = tgt_embeddings.size();
}

void GauravsModel::ReadSource(string filename) {
  ifstream f(filename);
  for (string line; getline(f, line);) {
    vector<string> pieces = tokenize(line, "|||");
    assert (pieces.size() == 2);
    assert (src_sentences.find(pieces[0]) == src_sentences.end());
    src_sentences[pieces[0]] = ReadSentence(pieces[1], &src_dict);
  }
  f.close();
}

Expression GauravsModel::GetRuleContext(const vector<int>& src, const string& tgt, LookupParameters& w_src, LookupParameters& w_tgt, ComputationGraph& cg, Model& cnn_model) {
  Expression i = getRNNRuleContext(src, tgt, &w_src, &w_tgt, cg, cnn_model);
  //FIXME: Austim
  return i;
  //return Expression(&cg, i);
}
