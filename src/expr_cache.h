#pragma once
#include <tuple>
#include <utility>
#include <unordered_map>
#include "cnn/expr.h"

using namespace std;
using namespace cnn::expr;

typedef tuple<unsigned, unsigned> cache_key_t;

struct cache_key_hash : public unary_function<cache_key_t, size_t> {
  size_t operator()(const cache_key_t& k) const {
    return get<0>(k) ^ get<1>(k);
  }
};

struct cache_key_equal : public binary_function<cache_key_t, cache_key_t, bool> {
  bool operator()(const cache_key_t& v0, const cache_key_t& v1) const {
    return (
      get<0>(v0) == get<0>(v1) && get<1>(v0) == get<1>(v1)
    );
  }
};


typedef unordered_map<const cache_key_t, Expression, cache_key_hash, cache_key_equal> cache_map_t;
typedef unordered_map<const cache_key_t, vector<Expression>, cache_key_hash, cache_key_equal> cache_map_vec_t;

struct ExpCache {
  cache_map_vec_t srcExpCache;
  cache_map_t tPhraseCache;
  cache_map_t sPhraseCache;
  unordered_map<unsigned, Expression> lContextCache;
  unordered_map<unsigned, Expression> rContextCache;
};
