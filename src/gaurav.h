class GauravsModel {
public:
  void ReadSource(string filename);
private:
  unordered_map<string, vector<int> > src_sentences;
  map<string, vector<float> > src_embeddings;
  map<string, vector<float> > tgt_embeddings;
  unsigned src_vocab_size;
  unsigned tgt_vocab_size;
  Dict src_dict;
  Dict tgt_dict;
};
