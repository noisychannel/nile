#include<regex>

#include "context.h"
#include "utils.h"

using namespace std;

vector<Context> getContext(const std::string& t, const vector<int>& s) {
  vector<Context> contextSeq;
  //std::vector<std::string> sParts = tokenize(s, " ");
  vector<int> sParts = s;
  vector<string> tParts = tokenize(t, " ");
  //sParts = strip(sParts);
  tParts = strip(tParts);
  // Unfortunate use of regex
  smatch sm;
  regex r("\\|(\\d+)-(\\d+)\\|");
  vector<string> currentTargetPhrase;
  for( vector<string>::const_iterator i = tParts.begin(); i != tParts.end(); ++i) {
    if (std::regex_match(*i, sm, r)) {
      // Match found : this is alignment information
      assert(sm.size() == 2);
      string tmpFrom = sm[0];
      string tmpTo = sm[1];
      unsigned srcFrom = atoi(tmpFrom.c_str());
      unsigned srcTo = atoi(tmpTo.c_str());
      // Get the source phrase
      unsigned srcId = 0;
      vector<int> leftContext;
      vector<int> rightContext;
      vector<int> sourcePhrase;
      for( vector<int>::const_iterator i = sParts.begin(); i != sParts.end(); ++i) {
        if (srcId < srcFrom) {
          //Generating left context
          leftContext.push_back(*i);
        }
        else if (srcId >= srcFrom && srcId <= srcTo) {
          // Generating source phrase
          sourcePhrase.push_back(*i);
        }
        else {
          // Generating right context
          rightContext.push_back(*i);
        }
        ++srcId;
      }

      //FIXME : This has to be re-written to use an int of target phrase stuff
      vector<int> tmpTargetPhrase;
      //vector<int> tmpTargetPhrase = ReadPhrase(currentTargetPhrase, &targetD);
      Context curContext = {leftContext, rightContext, sourcePhrase, tmpTargetPhrase};
      contextSeq.push_back(curContext);
      // Reset target token collector
      vector<string> currentTargetPhrase;
    }
    else {
      // Target tokens
      // Accumulate these till we see alignment info
      currentTargetPhrase.push_back(*i);
    }
  }
  return contextSeq;
}
