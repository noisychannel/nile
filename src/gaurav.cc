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

GauravsModel::GauravsModel(Model& cnn_model, string src_filename, string src_embedding_filename, string tgt_embedding_filename) {
  unordered_map<string, unsigned> tmp_src_dict, tmp_tgt_dict;
  unordered_map<unsigned, vector<float> > src_embedding_dict = LoadEmbeddings(src_embedding_filename, tmp_src_dict);
  unordered_map<unsigned, vector<float> > tgt_embedding_dict = LoadEmbeddings(tgt_embedding_filename, tmp_tgt_dict);
  cerr << "source embeddings contain " << src_embedding_dict.size() << " words with dimensionality " << src_embedding_dict.begin()->second.size() << "." << endl;
  cerr << "target embeddings contain " << tgt_embedding_dict.size() << " words with dimensionality " << tgt_embedding_dict.begin()->second.size() << "." << endl;
  assert (src_embedding_dict.size() > 0);
  assert (tgt_embedding_dict.size() > 0);
  assert (src_embedding_dict.begin()->second.size() > 0);
  assert (src_embedding_dict.begin()->second.size() == tgt_embedding_dict.begin()->second.size());

  unsigned embedding_dimensions = src_embedding_dict.begin()->second.size();
  src_vocab_size = src_embedding_dict.size();
  tgt_vocab_size = tgt_embedding_dict.size();

  src_embeddings = cnn_model.add_lookup_parameters(src_vocab_size, {embedding_dimensions});
  tgt_embeddings = cnn_model.add_lookup_parameters(tgt_vocab_size, {embedding_dimensions});

  for(unsigned i = 0; i < src_embedding_dict.size(); ++i) {
    src_embeddings->Initialize(i, src_embedding_dict[i]);
  }

  for(unsigned i = 0; i < tgt_dict.size(); ++i) {
    tgt_embeddings->Initialize(i, tgt_embedding_dict[i]);
  }

  ReadSource(src_filename);
}

void GauravsModel::ReadSource(string filename) {
  ifstream f(filename);
  for (string line; getline(f, line);) {
    vector<string> pieces = tokenize(line, "|||");
    assert (pieces.size() == 2);
    assert (src_sentences.find(pieces[0]) == src_sentences.end());
    vector<int> temp = ReadSentence(pieces[1], &src_dict);
    vector<unsigned> utemp(temp.size());
    for (unsigned i = 0; i < temp.size(); ++i) {
      assert (temp[i] >= 0);
      utemp[i] = (unsigned)temp[i];
    }
    src_sentences[pieces[0]] = utemp;
  }
  f.close();
}

Expression GauravsModel::GetRuleContext(const vector<unsigned>& src, const vector<unsigned>& tgt, const vector<PhraseAlignmentLink>& alignment, ComputationGraph& cg, Model& cnn_model) {
  return getRNNRuleContext(src, tgt, alignment, src_embeddings, tgt_embeddings, cg, cnn_model);
}

vector<unsigned> GauravsModel::ConvertTargetSentence(const vector<string>& words) {
  vector<unsigned> r (words.size());
  for (unsigned i = 0; i < words.size(); ++i) {
    r[i] = tgt_dict.Convert(words[i]);
  }
  return r;
}

vector<unsigned> GauravsModel::GetSourceSentence(const string& sent_id) {
  return src_sentences[sent_id];
}
