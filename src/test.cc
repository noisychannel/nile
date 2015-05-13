#include "embedding.h"
#include <iostream>
#include <sstream>
#include <map>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
  std::string fileName = "/export/a04/gkumar/code/custom/brae/tools/word2vec/fisher_en.vectors.50.sg.bin";
  const char *fileCharName = fileName.c_str();
  struct Embedding allEmbeddings = load(fileCharName);
  std::map<string, double> embedDict;

  for (int i = 0; i < allEmbeddings.words; i++){
    std::string word = &allEmbeddings.vocab[i * max_w];
    std::vector<float> wordEmbedding;

    for (int b = 0; b< allEmbeddings.size; b++) {
      wordEmbedding.push_back(allEmbeddings.M[i * allEmbeddings.size + b]); 
    }
    cerr << word << endl;
    for( std::vector<float>::const_iterator i = wordEmbedding.begin(); i != wordEmbedding.end(); ++i) {

      cerr << *i << " ";
    }
    embedDict[word] = 1;
  }
}
