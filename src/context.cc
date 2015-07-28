#include<regex>
#include<tuple>
#include<functional>

#include "context.h"
#include "utils.h"

using namespace std;

vector<Context> getContext(const vector<unsigned>& src, const vector<unsigned>& tgt,
                            const vector<PhraseAlignmentLink>& links) {

  vector<Context> contextSeq;
  vector<vector<double> > coverageVectors;

  for (unsigned i = 0; i < links.size(); ++i) {
    vector<double> cv;
    if (coverageVectors.size() == 0) {
      cv.resize(src.size(), 0.0);
    }
    else {
      cv = coverageVectors.back();
    }
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
        // Update coverage vector
        cv[src_id] = 1.0;
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

    // FIXME : Debugging info
    //cerr << currentSrcSpan.src_start << " " << currentSrcSpan.src_end << endl;
    //cerr << currentSrcSpan.tgt_start << " " << currentSrcSpan.tgt_end << endl;
    //for (auto cv_it = cv.begin(); cv_it != cv.end(); ++cv_it) {
      //cerr << *cv_it;
    //}
    //cerr << endl;
    //Push cv to all coverage vectors for this derivation
    coverageVectors.push_back(cv);

    //cerr << " ******** " << endl;
    Context curContext = {leftContext, rightContext, sourcePhrase, targetPhrase,
                          make_pair(currentSrcSpan.src_start, currentSrcSpan.src_end),
                          make_pair(currentSrcSpan.tgt_start, currentSrcSpan.tgt_end)};
    contextSeq.push_back(curContext);
    assert (curContext.leftContext.size() == contextSeq.back().leftContext.size());
  }
  //abort();

  return contextSeq;
}
