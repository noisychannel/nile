#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "utils.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>
#include <unordered_set>
#include <unordered_map>
#include <climits>
#include <csignal>

#include "kbestlist.h"
#include "reranker.h"
#include "feature_extractor.h"
#include "dataview.h"

using namespace std;
using namespace cnn;
namespace po = boost::program_options;

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
    cerr << "Usage: " << program_name << " model kbest.txt" << endl;
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

int main(int argc, char** argv) {
  po::options_description desc("description");
  desc.add_options()
  ("kbest_filename", po::value<string>()->required(), "Input k-best hypothesis file")
  ("model_filename", po::value<string>()->required(), "Reranker model")
  ("source_filename", po::value<string>()->default_value(""), "(Optional) List of source sentences corresponding to the input k-best list. Only required if using Gaurav's Model")
  ("help", "Display this help message");

  po::positional_options_description positional_options;
  positional_options.add("kbest_filename", 1);
  positional_options.add("model_filename", 1);
  positional_options.add("source_filename", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

 if (vm.count("help")) {
    cerr << desc;
    ShowUsageAndExit(argv[0]);
    return 1;
  }

  po::notify(vm);

  signal (SIGINT, ctrlc_handler);
  const string model_filename = vm["model_filename"].as<string>();
  const string kbest_filename = vm["kbest_filename"].as<string>();
  const string source_filename = vm["source_filename"].as<string>();

  cerr << "Building model...\n"; 
  cnn::Initialize(argc, argv);

  RerankerModel* reranker_model = NULL;
  KbestList* kbest_list = new KbestListInRam(kbest_filename);
  KbestListDataView* data_view = NULL;
  KbestFeatureExtractor* feature_extractor = NULL;
  Model cnn_model;

  cerr << "Reading model...\n";
  ifstream model_file(model_filename);
  boost::archive::text_iarchive ia(model_file);
  ia >> reranker_model;
  ia >> data_view;
  ia >> feature_extractor;
  ia >> cnn_model;

  vector<KbestHypothesis> hypotheses;
  vector<vector<float> > hypothesis_features(hypotheses.size());
  vector<float> metric_scores(hypotheses.size()); //unused

  unsigned num_sentences = 0;
  /*while (kbest_list->NextSet(hypotheses)) {
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
  }*/

  if (kbest_list != NULL) {
    delete kbest_list;
    kbest_list = NULL;
  }
 
  return 0;
}
