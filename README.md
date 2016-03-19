# csm
Context-Sensitive Translation with Non-Linear Rerankers

#### Important: Eigen version requirement

You need the [development version of the Eigen library](https://bitbucket.org/eigen/eigen) for this software to function. **If you use any of the released versions, you may get assertion failures or compile errors.**

#### Installing

Make sure you have [`cnn`](https://github.com/clab/cnn) installed.
Also ensure that you have the [development version of the Eigen library](https://bitbucket.org/eigen/eigen) available.

Locate the necessary libraries for us:

    ln -s [PATH_TO_CNN] ./.cnn
    ln -s [PATH_TO_EIGEN] ./.eigen

Then to compile, run

    make -j 2

#### Training Models

To train a non-linear reranker on the dense features, run:

./bin/train [k-best-file] > model_file

To train a non-linear reranker on the dense features with the context sensitive features, run:

./bin/train [k-best-file] --context [SOURCE_EMBEDDINGS] [TARGET_EMBEDDINGS] [SOURCE_SENTENCES] > model_file

To decode (rerank) with a pre-trained model, run

./bin/rerank [MODEL] [k-best-file] [SOURCE_SENTENCES]

Many further options are available are training. To find out more, run

./bin/train --help
