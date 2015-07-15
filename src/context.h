#include<vector>
#include<string>

using namespace std;

struct Context {
  const vector<int>& leftContext;
  const vector<int>& rightContext;
  const vector<int>& sourceRule;
  const vector<int>& targetRule;
};

vector<Context> getContext(const string& t, const vector<int>& s);
