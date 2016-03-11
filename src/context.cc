#include<regex>
#include<tuple>
#include<functional>

#include "context.h"
#include "utils.h"

using namespace std;

vector<Context> getContext(const vector<unsigned>& src, const vector<unsigned>& tgt,
                            const vector<PhraseAlignmentLink>& links) {

  vector<Context> contextSeq;

  for (unsigned i = 0; i < links.size(); ++i) {
    vector<double> cv;
    PhraseAlignmentLink currentSrcSpan = links[i];
    vector<unsigned> leftContext;
    vector<unsigned> rightContext;
    vector<unsigned> sourcePhrase;
    vector<unsigned> targetPhrase;
    auto sourceIdx = make_tuple(currentSrcSpan.src_start, currentSrcSpan.src_end);
    //cerr << "(" << currentSrcSpan.src_start << "," << currentSrcSpan.src_end << ") (" << currentSrcSpan.tgt_start << "," << currentSrcSpan.tgt_end << ")" << endl;
    for (unsigned src_id = 0; src_id < src.size(); ++src_id) {
      if (src_id < currentSrcSpan.src_start) {
        //Generating left context
        leftContext.push_back(src[src_id]);
      }
      else if (src_id >= currentSrcSpan.src_start && src_id < currentSrcSpan.src_end) {
        // Generating source phrase
        sourcePhrase.push_back(src[src_id]);
      }
      else if (src_id >= currentSrcSpan.src_end){
        // Generating right context
        rightContext.push_back(src[src_id]);
      }
    }

    // Iterate over target
    for ( auto tgt_it = tgt.begin() + currentSrcSpan.tgt_start;
          tgt_it != tgt.begin() + currentSrcSpan.tgt_end; ++tgt_it) {
      targetPhrase.push_back(*tgt_it);
    }

    // Reverse the right context before storing it
    // The right context RNN is a backward one!
    reverse(rightContext.begin(), rightContext.end());

    //Create context object
    //Add coverage vector to the context
    Context curContext = {leftContext, rightContext, sourcePhrase, targetPhrase,
                          make_pair(currentSrcSpan.src_start, currentSrcSpan.src_end),
                          make_pair(currentSrcSpan.tgt_start, currentSrcSpan.tgt_end),
                          cv};

    contextSeq.push_back(curContext);
    assert (curContext.leftContext.size() == contextSeq.back().leftContext.size());
  }

  return contextSeq;
}
