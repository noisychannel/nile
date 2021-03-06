#include "utils.h"

vector<string> tokenize(const string& input, const string& delimiter, int max_times) {
  vector<string> tokens;
  //tokens.reserve(max_times);
  size_t last = 0;
  size_t next = 0;
  while ((next = input.find(delimiter, last)) != string::npos && tokens.size() < max_times) {
    tokens.push_back(input.substr(last, next-last));
    last = next + delimiter.length();
  }
  tokens.push_back(input.substr(last));
  return tokens;
}

vector<string> tokenize(const string& input, const string& delimiter) {
  return tokenize(input, delimiter, input.length());
}

vector<string> tokenize(const string& input, char delimiter) {
  return tokenize(input, string(1, delimiter));
}

string strip(const string& input) {
  size_t start = 0;
  size_t end = input.length();

  for (; start < input.length(); ++start) {
    if (!isspace(input[start])) {
      break;
    }
  }
 
  for (; end > 0; --end) {
    if (!isspace(input[end - 1])) {
      break;
    }
  }

  if (end <= start) {
    return "";
  }

  return input.substr(start, end - start);
}

vector<string> strip(const vector<string>& input) {
  vector<string> output(input.size());
  for (unsigned i = 0; i < input.size(); ++i) {
    output[i] = strip(input[i]);
  }
  return output;
}

map<string, double> parse_feature_string(const string& input) {
  map<string, double> output;
  for (string piece : tokenize(input, " ")) {
    vector<string> kvp = tokenize(piece, "=", 1);
    if (kvp.size() != 2) {
      cerr << "Invalid feature name-value pair: \"" << piece << "\n";
      //cerr << "Invalid feature name-value pair: \"" << boost::algorithm::join(kvp, "=") << "\n";
      exit(1);
    }

    string name = kvp[0];
    double value = std::stod(kvp[1]);
    output[name] = value;
  }
  return output;
}

vector<float> itobit(int x, int vec_len) {
  std::vector<float> ret;
  while(x) {
    if (x & 1)
      ret.push_back(1.0);
    else
      ret.push_back(0.0);
    x >>= 1;
  }

  reverse(ret.begin(), ret.end());

  while (ret.size() < vec_len) {
    ret.push_back(0.0);
  }
  return ret;
}

map<int, vector<float> > create_bit_vector_cache(int start, int end, int vec_len) {
  map<int, vector<float> > bit_cache;
  for (int i = start; i <= end; ++i) {
    bit_cache.insert(make_pair(i, itobit(i, vec_len)));
  }
  return bit_cache;
}
