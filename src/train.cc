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
#include "feature_extractor.h"

#define SAFE_DELETE(p) if ((p) != NULL) { delete (p); (p) = NULL; }

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

RerankerModel* CreateRerankerModel(Model& cnn_model, unsigned num_dimensions, const po::variables_map& vm) {
  const unsigned hidden_size = vm["hidden_size"].as<unsigned>();
  RerankerModel* reranker_model = NULL;
  if (hidden_size > 0) {
    reranker_model = new NonlinearRerankerModel(&cnn_model, num_dimensions, hidden_size);
  }
  else {
    reranker_model = new LinearRerankerModel(&cnn_model, num_dimensions);
  }
  return reranker_model;
}

void Serialize(const RerankerModel* reranker_model, const KbestListInRamDataView* data_view, const KbestFeatureExtractor* feature_extractor, const Model& cnn_model) {
  ftruncate(fileno(stdout), 0);
  fseek(stdout, 0, SEEK_SET);
  boost::archive::text_oarchive oa(cout);
  oa << reranker_model;
  oa << data_view;
  oa << feature_extractor;
  oa << cnn_model;
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
  ("gaurav", po::value<vector<string> >()->multitoken(), "Use Gaurav's crazy-ass model. Specify source sentences, source embeddings, target embeddings.")
  ("combined", "Use the normal model in addition to Gaurav's. Specify --gaurav with the necessary files in addition to this flag.")
  ("gaurav_mlp", "Use the concat-MLP variation instead of the default behavior which sums the context vectors from the model components.")
  ("ebleu", "Use ebleu loss function (default)")
  ("pro", "Use pro loss function (100 samples)")
  ("1vsrest", "Use 1-vs-rest loss function")
  ("help", "Display this help message");

  po::positional_options_description positional_options;
  positional_options.add("kbest_filename", 1);
  positional_options.add("dev_filename", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    ShowUsageAndExit(argv[0]);
    return 1;
  }

  po::notify(vm);

  if (vm.count("gaurav")) {
    vector<string> gaurav_files = vm["gaurav"].as<vector<string> >();
    if (gaurav_files.size() < 3) {
      cerr << "Gaurav's model requires these files: source_sentences, source_embeddings, target_embeddings [, dev_source]" << endl;
      return 1;
    }
  }

  bool use_concat_mlp = false;
  if (vm.count("ebleu")) {
    use_concat_mlp = true;
  }

  const int kEBLEU = 0;
  const int kPRO = 1;
  const int k1VSREST = 2;
  int loss_function = kEBLEU;
  if (vm.count("ebleu")) {
    loss_function = kEBLEU;
  }
  else if (vm.count("pro")) {
    loss_function = kPRO;
  }
  else if (vm.count("1vsrest")) {
    loss_function = k1VSREST;
  }

  const string kbest_filename = vm["kbest_filename"].as<string>();
  const string dev_filename = vm["dev_filename"].as<string>();
  const unsigned max_features = vm["max_features"].as<unsigned>();
  const unsigned num_iterations = vm["num_iterations"].as<unsigned>();
  Model cnn_model;

  cerr << "Running on " << Eigen::nbThreads() << " threads." << endl;

  cnn::Initialize(argc, argv);
  KbestListInRam* train_kbest_list = NULL;
  KbestListInRam* dev_kbest_list = NULL;
  KbestListInRamDataView* train_data_view = NULL;
  KbestListInRamDataView* dev_data_view = NULL;
  KbestFeatureExtractor* train_feature_extractor = NULL; // These two feature extractors should be the same
  KbestFeatureExtractor* dev_feature_extractor = NULL;

  train_kbest_list = new KbestListInRam(kbest_filename);
  if (vm.count("gaurav") > 0) {
    vector<string> gauravs_shit = vm["gaurav"].as<vector<string> >();
    string source_file = gauravs_shit[0];
    string source_embeddings_file = gauravs_shit[1];
    string target_embeddings_file = gauravs_shit[2];
    if (vm.count("combined") > 0) {
      train_data_view = new CombinedDataView(train_kbest_list, source_file);
      train_feature_extractor = new CombinedFeatureExtractor(dynamic_cast<CombinedDataView*>(train_data_view), cnn_model, source_embeddings_file, target_embeddings_file, use_concat_mlp);
    }
    else {
      train_data_view = new GauravDataView(train_kbest_list, source_file);
      train_feature_extractor = new GauravsFeatureExtractor(dynamic_cast<GauravDataView*>(train_data_view), cnn_model, source_embeddings_file, target_embeddings_file, use_concat_mlp);
    }
  }
  else {
    train_data_view = new SimpleDataView(train_kbest_list, max_features);
    train_feature_extractor = new SimpleKbestFeatureExtractor(dynamic_cast<SimpleDataView*>(train_data_view));
  }

  if (dev_filename.length() > 0) {
    dev_kbest_list = new KbestListInRam(dev_filename);
    if (vm.count("gaurav") > 0) {
      vector<string> gauravs_shit = vm["gaurav"].as<vector<string> >();
      assert (gauravs_shit.size() >= 4);
      string source_file = gauravs_shit[3];
      string source_embeddings_file = gauravs_shit[1];
      string target_embeddings_file = gauravs_shit[2];
      if (vm.count("combined") > 0) {
        dev_data_view = new CombinedDataView(dev_kbest_list, source_file);
        dev_feature_extractor = new CombinedFeatureExtractor(dynamic_cast<CombinedDataView*>(dev_data_view), dynamic_cast<CombinedFeatureExtractor*>(train_feature_extractor));
      }
      else {
        dev_data_view = new GauravDataView(dev_kbest_list, source_file);
        dev_feature_extractor = new GauravsFeatureExtractor(dynamic_cast<GauravDataView*>(dev_data_view), dynamic_cast<GauravsFeatureExtractor*>(train_feature_extractor));
      }
    }
    else {
      dev_data_view = new SimpleDataView(dev_kbest_list, dynamic_cast<SimpleDataView*>(train_data_view));
      dev_feature_extractor = new SimpleKbestFeatureExtractor(dynamic_cast<SimpleDataView*>(dev_data_view));
    }
  }

  RerankerModel* reranker_model = CreateRerankerModel(cnn_model, train_feature_extractor->num_dimensions(), vm);
  Trainer* trainer = CreateTrainer(cnn_model, vm);

  cerr << "Training model...\n";
  double dev_score = 0.0;
  double best_dev_score = 0.0;
 
  for (unsigned iteration = 0; iteration <= num_iterations; iteration++) {
    double loss = 0.0;
    unsigned num_sentences = 0;
    train_feature_extractor->Reset();
  
    while (train_feature_extractor->MoveToNextSentence()) {
      num_sentences++;
      cerr << num_sentences << "\r";
      unsigned sent_index = num_sentences - 1;
    
      unsigned best = train_kbest_list->BestHypIndex(sent_index);
      ComputationGraph cg;

      vector<Expression> metric_scores;
      vector<Expression> model_scores;
      for (unsigned hyp_index = 0; train_feature_extractor->MoveToNextHypothesis(); ++hyp_index) {
        Expression hypothesis_features = train_feature_extractor->GetFeatures(cg);
        metric_scores.push_back(train_feature_extractor->GetMetricScore(cg));
        Expression model_score = reranker_model->score(hypothesis_features, cg);
        model_scores.push_back(model_score);
      }

      // EBLEU
      if (loss_function == kEBLEU) {
        Expression hyp_probs = softmax(concatenate(model_scores));
        Expression metric_score_vector = concatenate(metric_scores);
        Expression ebleu = dot_product(hyp_probs, metric_score_vector);
        Expression final = -ebleu;
        //reranker_model->BuildComputationGraph(hypothesis_features, metric_scores, cg);
      }
      // 1-vs-rest
      else if (loss_function == k1VSREST) {
        Expression hyp_probs = softmax(concatenate(model_scores));
        Expression loss = hinge(hyp_probs, &best, 0.1);
      }
      // PRO
      else if (loss_function == kPRO) {
        if (model_scores.size() < 2) {
          continue;
        }
        vector<Expression> losses;
        for (unsigned i = 0, j = 0; i < 100; ++i, ++j) {
          unsigned a = rand() % metric_scores.size();
          unsigned b = rand() % metric_scores.size();
          int comp = train_kbest_list->CompareHyps(sent_index, a, b);
          unsigned good, bad;
          if (comp < 0) {
            good = b;
            bad = a;
          }
          else if (comp > 0) {
            good = a;
            bad = b;
          }
          else {
            i -= 1;
            if (j > 100 * i && i > 100) {
              assert (false && "Unable to find good hypothesis pairs!");
            }
            continue;
          }

          Expression loss = 1.0 - (model_scores[bad] - model_scores[good]);
          losses.push_back(loss);
        }
        Expression loss = sum(losses);
      }
      else {
        assert (false);
      }

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
      assert (dev_kbest_list != NULL);
      dev_feature_extractor->Reset();
      while (dev_feature_extractor->MoveToNextSentence()) {
        dev_sentences++;
        vector<Expression> hypothesis_features;
        vector<Expression> metric_scores;
        ComputationGraph cg; 
        for (; dev_feature_extractor->MoveToNextHypothesis(); ) {
          hypothesis_features.push_back(dev_feature_extractor->GetFeatures(cg));
          metric_scores.push_back(dev_feature_extractor->GetMetricScore(cg));
        }
        reranker_model->BatchScore(hypothesis_features, cg);
        vector<float> scores = as_vector(cg.incremental_forward());
        concatenate(metric_scores);
        vector<float> m = as_vector(cg.incremental_forward());
        unsigned best_index = argmax(scores);
        Expression best_score_expr = pick(concatenate(metric_scores), best_index);
        dev_score += as_scalar(cg.incremental_forward());
      }
      assert (dev_sentences > 0);
      dev_score /= dev_sentences;
      cerr << "Dev score: " << dev_score;
      bool new_best = (dev_score > best_dev_score);
      if (new_best) {
        best_dev_score = dev_score;
        cerr << " (New best!)";
      }
      cerr << endl;
      if (new_best) {
        Serialize(reranker_model, train_data_view, train_feature_extractor, cnn_model);
      }
    }
  }

  if (dev_filename.length() == 0) {
    Serialize(reranker_model, train_data_view, train_feature_extractor, cnn_model);
  }

  SAFE_DELETE(train_feature_extractor);
  SAFE_DELETE(train_data_view);
  SAFE_DELETE(train_kbest_list);
  SAFE_DELETE(dev_feature_extractor);
  SAFE_DELETE(dev_data_view);
  SAFE_DELETE(dev_kbest_list);
  SAFE_DELETE(reranker_model);
  SAFE_DELETE(trainer);

  return 0;
}
