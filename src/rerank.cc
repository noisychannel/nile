#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "utils.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>

#include <iostream>
#include <fstream>
#include <unordered_set>
#include <unordered_map>
#include <climits>
#include <csignal>

#include "kbestlist.h"
#include "reranker.h"
#include "kbest_converter.h"

using namespace std;
using namespace cnn;

bool ctrlc_pressed = false;
void ctrlc_handler(int signal) {
  if (ctrlc_pressed) {
    exit(1);
  }
  else {
    ctrlc_pressed = true;
  }
}

int main(int argc, char** argv) {
  if (argc < 3) {
    cerr << "Usage: " << argv[0] << " model kbest.txt" << endl;
    cerr << endl;
    cerr << "Where kbest.txt contains lines of them form" << endl;
    cerr << "sentence_id ||| hypothesis ||| features ||| ... " << endl;
    cerr << "The first three fields must be the sentence id, hypothesis, and features." << endl;
    cerr << "Any extra fields are ignored." << endl;
    cerr << endl;
    cerr << "Here's an example of a valid input line:" << endl;
    cerr << "0 ||| <s> ovatko ne syyt tai ? </s> ||| MaxLexEgivenF=1.26902 Glue=2 LanguageModel=-14.2355 SampleCountF=9.91427 ||| -1.32408" << endl;
    exit(1);
  }
  signal (SIGINT, ctrlc_handler);
  const string model_filename = argv[1];
  const string kbest_filename = argv[2];

  cerr << "Building model...\n"; 
  cnn::Initialize(argc, argv);

  KbestConverter* converter = NULL;
  RerankerModel* reranker_model = NULL;

  cerr << "Reading model...\n";
  ifstream model_file(model_filename);
  boost::archive::text_iarchive ia(model_file);
  ia >> converter;
  ia >> reranker_model;

  vector<KbestHypothesis> hypotheses;
  vector<vector<float> > hypothesis_features(hypotheses.size());
  vector<float> metric_scores(hypotheses.size()); //unused

  unsigned num_sentences = 0;
  KbestList* kbest_list = new KbestListInRam(kbest_filename);
  while (kbest_list->NextSet(hypotheses)) {
    assert (hypotheses.size() > 0);
    num_sentences++;
    cerr << num_sentences << "\r";

    ComputationGraph cg;
    converter->ConvertKbestSet(hypotheses, hypothesis_features, metric_scores);
    KbestHypothesis* best = NULL;
    double best_score = 0.0;
    for (unsigned i = 0; i < hypotheses.size(); ++i) {
      reranker_model->score(&hypothesis_features[i], cg);
      double score = as_scalar(cg.incremental_forward());
      if (score > best_score || best == NULL) {
        best = &hypotheses[i];
        best_score = score;
      }
    }
    cout << best->sentence_id << " ||| " << best->sentence << " ||| ";
    cout << "features yay" << " ||| " << best->metric_score << endl;
  }

  if (kbest_list != NULL) {
    delete kbest_list;
    kbest_list = NULL;
  }

  if (converter != NULL) {
    delete converter;
    converter = NULL;
  }

  if (reranker_model != NULL) {
    delete reranker_model;
    reranker_model = NULL;
  }
 
  return 0;
}
