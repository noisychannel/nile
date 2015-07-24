#pragma once
#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#include <cassert>
#include <cctype>
#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/regex.hpp>

using namespace std;

vector<string> tokenize(const string& input, const string& delimiter, int max_times);
vector<string> tokenize(const string& input, const string& delimiter);
vector<string> tokenize(const string& input, char delimiter);

string strip(const string& input);
vector<string> strip(const vector<string>& input);

map<string, double> parse_feature_string(const string& input);

vector<float> itobit(int x, int vec_len);

map<int, vector<float> > create_bit_vector_cache(int start, int end, int vec_len);
