#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/expr.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/export.hpp>

#include <iostream>
#include <fstream>
#include <unordered_set>
#include <unordered_map>
#include <climits>
#include <csignal>

#include "kbestlist.h"
#include "utils.h"
#include "reranker.h"

#define NONLINEAR

using namespace std;
using namespace cnn;
using namespace cnn::expr;

const unsigned num_iterations = 100;
const unsigned max_features = 1000;
const unsigned hidden_size = 50;
const bool nonlinear = true;

bool ctrlc_pressed = false;
void ctrlc_handler(int signal) {
  if (ctrlc_pressed) {
    exit(1);
  }
  else {
    ctrlc_pressed = true;
  }
}

void ShowUsageAndExit(string program_name) {
    cerr << "Usage: " << program_name << " kbest.txt [dev.txt]" << endl;
    cerr << endl;
    cerr << "Where kbest.txt contains lines of them form" << endl;
    cerr << "sentence_id ||| hypothesis ||| features ||| ... ||| metric score" << endl;
    cerr << "The first three fields must be the sentence id, hypothesis, and features." << endl;
    cerr << "The last field must be the metric score of each hypothesis." << endl;
    cerr << "Any fields in between are ignored." << endl;
    cerr << endl;
    cerr << "Here's an example of a valid input line:" << endl;
    cerr << "0 ||| <s> ovatko ne syyt tai ? </s> ||| MaxLexEgivenF=1.26902 Glue=2 LanguageModel=-14.2355 SampleCountF=9.91427 ||| -1.32408 ||| 21.3" << endl;
    exit(1);
}

int main(int argc, char** argv) {
  signal (SIGINT, ctrlc_handler);
  if (argc < 2) {
    ShowUsageAndExit(argv[0]);
  }
  const string kbest_filename = argv[1];
  const string dev_filename = (argc >= 3) ? argv[2] : "";

  cerr << "Running on " << Eigen::nbThreads() << " threads." << endl;

  cnn::Initialize(argc, argv);
  Model cnn_model;
  //SimpleSGDTrainer sgd(&cnn_model, 0.0, 0.1);
  AdadeltaTrainer sgd(&cnn_model, 0.0, 1e-6);
  sgd.eta_decay = 0.05;

  RerankerModel* reranker_model = NULL;
  if (nonlinear) {
    reranker_model = new NonlinearRerankerModel(hidden_size);
  }
  else {
    reranker_model = new LinearRerankerModel();
  }

  reranker_model->ReadFeatureNames(kbest_filename, max_features);
  reranker_model->InitializeParameters(cnn_model);

  cerr << "Training model...\n";
  vector<KbestHypothesis> hypotheses;
  vector<vector<float> > hypothesis_features(hypotheses.size());
  vector<float> metric_scores(hypotheses.size());

  for (unsigned iteration = 0; iteration <= num_iterations; iteration++) {
    double loss = 0.0;
    unsigned num_sentences = 0;
    KbestList kbest_list(kbest_filename);
    while (kbest_list.NextSet(hypotheses)) {
      assert (hypotheses.size() > 0);
      num_sentences++;
      cerr << num_sentences << "\r";

      ComputationGraph cg;
      reranker_model->ConvertKbestSet(hypotheses, hypothesis_features, metric_scores);
      reranker_model->BuildComputationGraph(hypothesis_features, metric_scores, cg);

      loss += as_scalar(cg.forward());
      if (iteration != 0) {
        cg.backward();
        sgd.update(1.0);
      }
      if (ctrlc_pressed) {
        break;
      }
    }
    if (ctrlc_pressed) {
      break;
    }
    if (dev_filename.length() > 0) {
      //double dev_score = score_devset();
    }
    cerr << "Iteration " << iteration << " loss: " << loss << " (EBLEU = " << -loss / num_sentences << ")" << endl;
  }

  boost::archive::text_oarchive oa(cout);
  oa << reranker_model;

  if (reranker_model != NULL) {
    delete reranker_model;
    reranker_model = NULL;
  }

  return 0;
}
