#pragma once
#include<vector>
#include<string>
#include "kbest_hypothesis.h"

using namespace std;

struct Context {
  const vector<unsigned> leftContext;
  const vector<unsigned> rightContext;
  const vector<unsigned> sourceRule;
  const vector<unsigned> targetRule;
  const tuple<unsigned, unsigned> srcIdx;
  const tuple<unsigned, unsigned> tgtIdx;
  vector<double> coverage;
};

vector<Context> getContext(const vector<unsigned>& src, const vector<unsigned>& tgt,
                            const vector<PhraseAlignmentLink>& links);
