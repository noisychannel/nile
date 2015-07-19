#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/expr.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/export.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>
#include <unordered_set>
#include <unordered_map>
#include <climits>
#include <csignal>

#include "kbestlist.h"
#include "utils.h"
#include "reranker.h"
#include "kbest_converter.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;
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

Trainer* CreateTrainer(Model& model, const po::variables_map& vm) {
  double regularization_strength = vm["regularization"].as<double>();
  double eta_decay = vm["eta_decay"].as<double>();
  bool clipping_enabled = (vm.count("no_clipping") == 0);
  unsigned learner_count = vm.count("sgd") + vm.count("momentum") + vm.count("adagrad") + vm.count("adadelta") + vm.count("rmsprop") + vm.count("adam");
  if (learner_count > 1) {
    cerr << "Invalid parameters: Please specify only one learner type.";
    exit(1);
  }

  Trainer* trainer = NULL;
  if (vm.count("momentum")) {
    double learning_rate = (vm.count("learning_rate")) ? vm["learning_rate"].as<double>() : 0.01;
    double momentum = vm["momentum"].as<double>();
    trainer = new MomentumSGDTrainer(&model, regularization_strength, learning_rate, momentum);
  }
  else if (vm.count("adagrad")) {
    double learning_rate = (vm.count("learning_rate")) ? vm["learning_rate"].as<double>() : 0.1;
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-20;
    trainer = new AdagradTrainer(&model, regularization_strength, learning_rate, eps);
  }
  else if (vm.count("adadelta")) {
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-6;
    double rho = (vm.count("rho")) ? vm["rho"].as<double>() : 0.95;
    trainer = new AdadeltaTrainer(&model, regularization_strength, eps, rho);
  }
  else if (vm.count("rmsprop")) {
    double learning_rate = (vm.count("learning_rate")) ? vm["learning_rate"].as<double>() : 0.1;
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-20;
    double rho = (vm.count("rho")) ? vm["rho"].as<double>() : 0.95;
    trainer = new RmsPropTrainer(&model, regularization_strength, learning_rate, eps, rho);
  }
  else if (vm.count("adam")) {
    double alpha = (vm.count("alpha")) ? vm["alpha"].as<double>() : 0.001;
    double beta1 = (vm.count("beta1")) ? vm["beta1"].as<double>() : 0.9;
    double beta2 = (vm.count("beta2")) ? vm["beta2"].as<double>() : 0.999;
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-8;
    trainer = new AdamTrainer(&model, regularization_strength, alpha, beta1, beta2, eps);
  }
  else { /* sgd */
    double learning_rate = (vm.count("learning_rate")) ? vm["learning_rate"].as<double>() : 0.1;
    trainer = new SimpleSGDTrainer(&model, regularization_strength, learning_rate);
  }
  assert (trainer != NULL);

  trainer->eta_decay = eta_decay;
  trainer->clipping_enabled = clipping_enabled;
  return trainer;
}

int main(int argc, char** argv) {
  signal (SIGINT, ctrlc_handler);

  po::options_description desc("description");
  desc.add_options()
  ("kbest_filename", po::value<string>()->required(), "Input k-best hypothesis file") 
  ("dev_filename", po::value<string>()->default_value(""), "(Optional) Dev k-best list, used for early stopping")
  ("sgd", "Use SGD for optimization")
  ("momentum", po::value<double>(), "Use SGD with this momentum value")
  ("adagrad", "Use Adagrad for optimization")
  ("adadelta", "Use Adadelta for optimization")
  ("rmsprop", "Use RMSProp for optimization")
  ("adam", "Use Adam for optimization")
  ("learning_rate,r", po::value<double>(), "Learning rate for optimizer (SGD, Adagrad, Adadelta, and RMSProp only)")
  ("alpha", po::value<double>(), "Alpha (Adam only)")
  ("beta1", po::value<double>(), "Beta1 (Adam only)")
  ("beta2", po::value<double>(), "Beta2 (Adam only)")
  ("rho", po::value<double>(), "Moving average decay parameter (RMSProp and Adadelta only)")
  ("epsilon", po::value<double>(), "Epsilon value for optimizer (Adagrad, Adadelta, RMSProp, and Adam only)")
  ("regularization", po::value<double>()->default_value(0.0), "L2 Regularization strength")
  ("eta_decay", po::value<double>()->default_value(0.05), "Learning rate decay rate (SGD only)")
  ("no_clipping", "Disable clipping of gradients")
  ("hidden_size,h", po::value<unsigned>()->default_value(0), "Hidden layer dimensionality. 0 = linear model")
  ("max_features", po::value<unsigned>()->default_value(UINT_MAX), "Maximum number of input features. Later features will be discarded.")
  ("num_iterations,i", po::value<unsigned>()->default_value(UINT_MAX), "Number of epochs to train for")
  ("help", "Display this help message");

  po::positional_options_description positional_options;
  positional_options.add("kbest_filename", 1);
  positional_options.add("dev_filename", 1);

  po::variables_map vm;
  //try {
    po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

    if (vm.count("help")) {
      cerr << desc; 
      ShowUsageAndExit(argv[0]);
      return 1;
    }

    po::notify(vm);
    
  //}
  /*catch (po::error& e) {
    cerr << "Error parsing arguments. Exiting..." << endl;
    return 1;
  }*/

  const string kbest_filename = vm["kbest_filename"].as<string>();
  const string dev_filename = vm["dev_filename"].as<string>();
  const unsigned hidden_size = vm["hidden_size"].as<unsigned>();
  const unsigned max_features = vm["max_features"].as<unsigned>();
  const unsigned num_iterations = vm["num_iterations"].as<unsigned>();

  cerr << "Running on " << Eigen::nbThreads() << " threads." << endl;

  cnn::Initialize(argc, argv);
  KbestConverter* converter = new KbestConverter(kbest_filename, max_features);

  RerankerModel* reranker_model = NULL;
  if (hidden_size > 0) {
    reranker_model = new NonlinearRerankerModel(converter->num_dimensions, hidden_size);
  }
  else {
    reranker_model = new LinearRerankerModel(converter->num_dimensions);
  }

  Trainer* trainer = CreateTrainer(reranker_model->cnn_model, vm);

  cerr << "Training model...\n";
  vector<KbestHypothesis> hypotheses;
  vector<vector<float> > hypothesis_features(hypotheses.size());
  vector<float> metric_scores(hypotheses.size());
  double dev_score = 0.0;
  double best_dev_score = 0.0;

  KbestList* kbest_list = new KbestListInRam(kbest_filename);
  KbestList* dev_kbest = NULL;
  if (dev_filename.length() > 0) {
    dev_kbest = new KbestListInRam(dev_filename);
  }

  for (unsigned iteration = 0; iteration <= num_iterations; iteration++) {
    double loss = 0.0;
    unsigned num_sentences = 0;
    kbest_list->Reset();

    while (kbest_list->NextSet(hypotheses)) {
      assert (hypotheses.size() > 0);
      num_sentences++;
      cerr << num_sentences << "\r";

      ComputationGraph cg;
      converter->ConvertKbestSet(hypotheses, hypothesis_features, metric_scores);
      reranker_model->BuildComputationGraph(hypothesis_features, metric_scores, cg);

      loss += as_scalar(cg.forward());
      if (iteration != 0) {
        cg.backward();
        trainer->update(1.0);
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
      dev_kbest->Reset();
      while (dev_kbest->NextSet(hypotheses)) {
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

  if (kbest_list == NULL) {
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

  if (trainer != NULL) {
    delete trainer;
    trainer = NULL;
  }

  return 0;
}
