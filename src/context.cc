#include<regex>

#include "context.h"
#include "utils.h"

using namespace std;

vector<Context> getContext(const vector<unsigned>& src, const vector<unsigned>& tgt,
                            const vector<PhraseAlignmentLink>& links) {

  vector<Context> contextSeq;
  //cerr << "Got source of length " << src.size() << endl;
  //cerr << "Got target of length " << tgt.size() << endl;
  for (unsigned i = 0; i < links.size(); ++i) {
    PhraseAlignmentLink currentSrcSpan = links[i];
    vector<unsigned> leftContext;
    vector<unsigned> rightContext;
    vector<unsigned> sourcePhrase;
    vector<unsigned> targetPhrase;
    //cerr << "(" << currentSrcSpan.src_start << "," << currentSrcSpan.src_end << ") (" << currentSrcSpan.tgt_start << "," << currentSrcSpan.tgt_end << ")" << endl;
    for (unsigned src_id = 0; src_id < src.size(); ++src_id) {
      if (src_id < currentSrcSpan.src_start) {
        //cerr << "Left" << endl;
        //Generating left context
        leftContext.push_back(src[src_id]);
      }
      else if (src_id >= currentSrcSpan.src_start && src_id < currentSrcSpan.src_end) {
        //cerr << "Phrase" << endl;
        // Generating source phrase
        sourcePhrase.push_back(src[src_id]);
      }
      else {
        //cerr << "Right" << endl;
        // Generating right context
        rightContext.push_back(src[src_id]);
      }
      //cerr << src_id << endl;
    }

    // Iterate over target
    for ( auto tgt_it = tgt.begin() + currentSrcSpan.tgt_start;
          tgt_it != tgt.begin() + currentSrcSpan.tgt_end; ++tgt_it) {
      targetPhrase.push_back(*tgt_it);
    }

    //cerr << leftContext.size() << endl;
    //cerr << rightContext.size() << endl;
    //cerr << sourcePhrase.size() << endl;
    //cerr << targetPhrase.size() << endl;
    //cerr << " ******** " << endl;
    Context curContext = {leftContext, rightContext, sourcePhrase, targetPhrase};
    //cerr << curContext.leftContext.size() << endl;
    //cerr << curContext.rightContext.size() << endl;
    //cerr << curContext.sourceRule.size() << endl;
    //cerr << curContext.targetRule.size() << endl;
    //cerr << " @@@@@@ " << endl;
    contextSeq.push_back(curContext);
    assert (curContext.leftContext.size() == contextSeq.back().leftContext.size());
  }

  return contextSeq;
}
