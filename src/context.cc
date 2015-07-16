#include<regex>

#include "context.h"
#include "utils.h"

using namespace std;

vector<Context> getContext(const vector<unsigned>& src, const vector<unsigned>& tgt,
                            const vector<PhraseAlignmentLink>& links) {

  vector<Context> contextSeq;
  for (unsigned i = 0; i < links.size(); ++i) {
    PhraseAlignmentLink currentSrcSpan = links[i];
    vector<unsigned> leftContext;
    vector<unsigned> rightContext;
    vector<unsigned> sourcePhrase;
    vector<unsigned> targetPhrase;
    unsigned srcId = 0;
    for ( auto src_it = src.begin(); src_it != src.end(); ++src_it ) {
      if (srcId < currentSrcSpan.src_start) {
        //Generating left context
        leftContext.push_back(*src_it);
      }
      else if (srcId >= currentSrcSpan.src_start && srcId < currentSrcSpan.src_end) {
        // Generating source phrase
        sourcePhrase.push_back(*src_it);
      }
      else {
        // Generating right context
        rightContext.push_back(*src_it);
      }
      ++srcId;
    }

    // Iterate over target
    for ( auto tgt_it = src.begin() + currentSrcSpan.tgt_start;
          tgt_it != src.begin() + currentSrcSpan.tgt_end; ++tgt_it) {
      targetPhrase.push_back(*tgt_it);
    }

    Context curContext = {leftContext, rightContext, sourcePhrase, targetPhrase};
    contextSeq.push_back(curContext);
  }

  return contextSeq;
}
