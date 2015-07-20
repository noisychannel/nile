#include "cnn/lstm.h"
#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/expr.h"
#include "cnn/dict.h"

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
#include "gaurav.h"
#include "kbest_converter.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

const unsigned max_features = 1000;
const unsigned num_iterations = 1000;
const unsigned hidden_size = 47;
const bool nonlinear = true;

unordered_map<string, vector<int> > ReadSource(string filename, Dict& src_dict) {
  unordered_map<string, vector<int> > r;
  ifstream f(filename);
  for (string line; getline(f, line);) {
    vector<string> pieces = tokenize(line, "|||");
    assert (pieces.size() == 2);
    assert (r.find(pieces[0]) == r.end());
    r[pieces[0]] = ReadSentence(pieces[1], &src_dict);
  }
  return r;
}

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
    cerr << "Usage: " << program_name << " tune_kbest tune_source source_embeddings target_embeddings" << endl;
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
  if (argc < 5) {
    ShowUsageAndExit(argv[0]);
  }
  const string kbest_filename = argv[1];
  const string source_filename = argv[2];
  const string source_embedding_filename = argv[3];
  const string target_embedding_filename = argv[4];

  cerr << "Running on " << Eigen::nbThreads() << " threads." << endl;

  cnn::Initialize(argc, argv);
  Model cnn_model;
  KbestConverter* converter = new KbestConverter(kbest_filename, max_features);

  GauravsModel gauravs_model (cnn_model, source_filename, source_embedding_filename, target_embedding_filename);

  RerankerModel* reranker_model = NULL;
  if (nonlinear) {
    reranker_model = new NonlinearRerankerModel(gauravs_model.OutputDimension(), hidden_size);
  }
  else {
    reranker_model = new LinearRerankerModel(gauravs_model.OutputDimension());
  } 

  //SimpleSGDTrainer sgd(&reranker_model->cnn_model, 0.0, 0.1);
  AdadeltaTrainer sgd(&reranker_model->cnn_model, 0.0);
  //sgd.eta_decay = 0.05;
  //sgd.clipping_enabled = false;

  cerr << "Training model...\n";
  vector<KbestHypothesis> hypotheses;
  vector<Expression> hypothesis_features;
  vector<float> metric_scores;

  KbestList* kbest_list = new KbestListInRam(kbest_filename);
  for (unsigned iteration = 0; iteration <= num_iterations; iteration++) {
    double loss = 0.0;
    unsigned num_sentences = 0;
    kbest_list->Reset();
    while (kbest_list->NextSet(hypotheses)) {
      assert (hypotheses.size() > 0);
      num_sentences++;
      cerr << num_sentences << "\r";

      hypothesis_features.clear();
      metric_scores.clear();

      ComputationGraph cg;
      for (KbestHypothesis& hyp : hypotheses) {
        vector<string> tgt_words;
        vector<PhraseAlignmentLink> alignment;
        converter->ConvertTargetString(hyp, tgt_words, alignment);
        vector<unsigned> src = gauravs_model.GetSourceSentence(hyp.sentence_id);
        vector<unsigned> tgt = gauravs_model.ConvertTargetSentence(tgt_words); 
        assert(src.size() > 0);
        assert(tgt.size() > 0);
        Expression hyp_features = gauravs_model.GetRuleContext(src, tgt, alignment, cg);
        hypothesis_features.push_back(hyp_features);
        metric_scores.push_back(hyp.metric_score);
      }
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
    cerr << "Gaurav Iteration " << iteration << " loss: " << loss << " (EBLEU = " << -loss / num_sentences << ")" << endl;
  }

  boost::archive::text_oarchive oa(cout);
  oa << converter;
  oa << reranker_model;

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
