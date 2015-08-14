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
    cerr << "Usage: " << program_name << " model kbest.txt [source.txt]" << endl;
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
  ("model_filename", po::value<string>()->required(), "Reranker model")
  ("kbest_filename", po::value<string>()->required(), "Input k-best hypothesis file")
  ("source_filename", po::value<string>()->default_value(""), "(Optional) List of source sentences corresponding to the input k-best list. Only required if using Gaurav's Model")
  ("help", "Display this help message");

  po::positional_options_description positional_options;
  positional_options.add("model_filename", 1);
  positional_options.add("kbest_filename", 1);
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
  KbestListInRam* kbest_list = new KbestListInRam(kbest_filename);
  KbestListInRamDataView* data_view = NULL;
  KbestFeatureExtractor* feature_extractor = NULL;
  Model cnn_model;

  cerr << "Reading model...\n";
  ifstream model_file(model_filename);
  if (!model_file.is_open()) {
    cerr << "ERROR: Unable to open model file " << model_filename << "!\n";
    return 1;
  }
  boost::archive::text_iarchive ia(model_file);
  ia >> reranker_model;
  ia >> data_view;
  ia >> feature_extractor; 
  //data_view->InitializeParameters(&cnn_model);
  data_view->Initialize(kbest_list, source_filename);
  feature_extractor->InitializeParameters(&cnn_model); 
  feature_extractor->SetDataPointer(data_view);
  reranker_model->InitializeParameters(&cnn_model);
  ia >> cnn_model;

  vector<KbestHypothesis> hypotheses;
  vector<vector<float> > hypothesis_features(hypotheses.size());
  vector<float> metric_scores(hypotheses.size()); //unused

  unsigned sent_index = 0;
  while (feature_extractor->MoveToNextSentence()) {
    cerr << sent_index << "\r";
    ComputationGraph cg;
    unsigned best_hyp_index = 0;
    double best_score = 0.0;
    unsigned hyp_index = 0;
    while (feature_extractor->MoveToNextHypothesis()) {
      Expression features = feature_extractor->GetFeatures(cg);
      reranker_model->score(features, cg);
      double score = as_scalar(cg.incremental_forward());
      if (score > best_score || hyp_index == 0) {
        best_hyp_index = hyp_index;
        best_score = score;
      }
      ++hyp_index;
    }
    assert (hyp_index > 0);
    const KbestHypothesis best = dynamic_cast<KbestListInRam*>(kbest_list)->Get(sent_index, best_hyp_index);
    cout << best.sentence_id << " ||| " << best.sentence << " ||| ";
    bool first = true;
    for (const auto& kvp : best.features) {
      if (!first) {
        cout << " ";
      }
      cout << kvp.first << "=" << kvp.second;
      first = false;
    }
    cout << " ||| " << best.metric_score << endl;

    sent_index++;
    cerr << sent_index << "\r";
  }

  if (kbest_list != NULL) {
    delete kbest_list;
    kbest_list = NULL;
  }
 
  return 0;
}
