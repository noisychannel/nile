#include <iostream>
#include <string>
#include "kbest_hypothesis.h"
using namespace std;

int main() {
  string line;
  while (getline(cin, line)) {
    KbestHypothesis hyp = KbestHypothesis::parse(line);
    vector<string> words;
    vector<PhraseAlignmentLink> links;
    words = hyp.TokenizedTarget(links);
    for (unsigned i = 0; i < words.size(); ++i) {
      cout << i << ": " << words[i] << endl;
    }
    for (unsigned i = 0; i < links.size(); ++i) {
      cout << "src=[" << links[i].src_start << ", " << links[i].src_end << "), tgt=[" << links[i].tgt_start << ", " << links[i].tgt_end << ")" << endl;
    }
    cout << endl;
  }
}
