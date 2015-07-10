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

vector<string> tokenize(string input, string delimiter, int max_times);
vector<string> tokenize(string input, string delimiter);
vector<string> tokenize(string input, char delimiter);

string strip(string input);
vector<string> strip(vector<string> input);

map<string, double> parse_feature_string(string input);
