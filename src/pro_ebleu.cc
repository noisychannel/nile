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

using namespace std;
using namespace cnn;
using namespace cnn::expr;

const unsigned num_iterations = 10000;
const unsigned max_features = 1000;
const unsigned hidden_size = 500;
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

template <typename T>
unsigned argmax(vector<T> v) {
  assert (v.size() > 0);
  T best = v[0];
  unsigned bi = 0;
  for (unsigned i = 1; i < v.size(); ++i) {
    if (v[i] > best) {
      bi = i;
      best = v[i];
    }
  }
  return bi;
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
  KbestConverter* converter = new KbestConverter(kbest_filename, max_features);

  RerankerModel* reranker_model = NULL;
  if (nonlinear) {
    reranker_model = new NonlinearRerankerModel(converter->num_dimensions, hidden_size);
  }
  else {
    reranker_model = new LinearRerankerModel(converter->num_dimensions);
  }

  //SimpleSGDTrainer sgd(&reranker_model->cnn_model, 0.0, 10.0);
  AdadeltaTrainer sgd(&reranker_model->cnn_model, 0.0);
  //sgd.eta_decay = 0.05;
  sgd.clipping_enabled = false;

  cerr << "Training model...\n";
  vector<KbestHypothesis> hypotheses;
  vector<vector<float> > hypothesis_features(hypotheses.size());
  vector<float> metric_scores(hypotheses.size());
  double dev_score = 0.0;
  double best_dev_score = 0.0;

  for (unsigned iteration = 0; iteration <= num_iterations; iteration++) {
    double loss = 0.0;
    unsigned num_sentences = 0;
    KbestList kbest_list(kbest_filename);
    while (kbest_list.NextSet(hypotheses)) {
      assert (hypotheses.size() > 0);
      num_sentences++;
      cerr << num_sentences << "\r";

      ComputationGraph cg;
      converter->ConvertKbestSet(hypotheses, hypothesis_features, metric_scores);
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
    cerr << "Iteration " << iteration << " loss: " << loss << " (EBLEU = " << -loss / num_sentences << ")" << endl;
    if (dev_filename.length() > 0) {
      dev_score = 0.0;
      unsigned dev_sentences = 0;
      KbestList dev_kbest(dev_filename);
      while (dev_kbest.NextSet(hypotheses)) {
        dev_sentences++;
        ComputationGraph cg;
        converter->ConvertKbestSet(hypotheses, hypothesis_features, metric_scores);
        reranker_model->BatchScore(hypothesis_features, metric_scores, cg);
        vector<float> scores = as_vector(cg.incremental_forward());
        unsigned best_index = argmax(scores);
        dev_score += metric_scores[best_index];
      }
      dev_score /= dev_sentences;
      cerr << "Dev score: " << dev_score;
      bool new_best = (dev_score > best_dev_score);
      if (new_best) {
        best_dev_score = dev_score;
        cerr << " (New best!)";
      }
      cerr << endl;
      if (new_best) {
        ftruncate(fileno(stdout), 0);
        fseek(stdout, 0, SEEK_SET);
        boost::archive::text_oarchive oa(cout);
        oa << converter;
        oa << reranker_model;
      }
    }
  }

  if (dev_filename.length() == 0) {
    boost::archive::text_oarchive oa(cout);
    oa << converter;
    oa << reranker_model;
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
