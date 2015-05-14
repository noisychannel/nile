//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

struct Embedding {
  char *vocab;
  float *M;
  long words;
  long size;
};

std::map<std::string, std::vector<float>> loadEmbeddings(const char* file_name) {
//int main(int argc, char **argv) {
  FILE *f;
  //char file_name[max_size];
  float len;
  long long words, size, a, b;
  char ch;
  float *M;
  char *vocab;
  //if (argc < 2) {
    //printf("Usage: ./distance <FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n");
    //return 0;
  //}
  //strcpy(file_name, argv[1]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    exit(1);
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    exit(1);
  }
  for (b = 0; b < words; b++) {
    fscanf(f, "%s%c", &vocab[b * max_w], &ch);
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
  }

  // Changes made by Gaurav Kumar (gkumar@cs.jhu.edu)
  // Returns a dictionary of word embeddings
  std::map<std::string, std::vector<float>> embedDict;
  for (int i = 0; i < words; i++){
    std::string word = std::string(&vocab[i * max_w]);
    std::vector<float> wordEmbedding;

    for (int b = 0; b< size; b++) {
      wordEmbedding.push_back(M[i * size + b]);
    }
    //cerr << word << endl;
    for( std::vector<float>::const_iterator i = wordEmbedding.begin(); i != wordEmbedding.end(); ++i) {
      std::cerr << *i << " ";
      embedDict[word].push_back(*i);;
    }
  }

  return embedDict;
}
