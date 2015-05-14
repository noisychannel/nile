CC=g++
CNN_DIR = /Users/austinma/git/cnn/
#CNN_DIR=/export/a04/gkumar/code/cnn/
#CNN_DIR=/Users/gaurav/Projects/cnn/
EIGEN = /Users/austinma/git/eigen
#EIGEN=/export/a04/gkumar/code/eigen/
#EIGEN=/Users/gaurav/Projects/eigen/
CNN_BUILD_DIR=$(CNN_DIR)/build
INCS=-I$(CNN_DIR) -I$(CNN_BUILD_DIR) -I$(EIGEN)
LIBS=-L$(CNN_BUILD_DIR)/cnn/
FINAL=-lcnn -lboost_regex -lboost_serialization
CFLAGS=-O3 -ffast-math -funroll-loops
BINDIR=bin
SRCDIR=src

.PHONY: clean
all: $(BINDIR)/pro $(BINDIR)/rerank $(BINDIR)/rnnContextRule
#all: $(BINDIR)/embedding $(BINDIR)/pro

#$(BINDIR)/embedding: $(SRCDIR)/embedding.cc
	#mkdir -p $(BINDIR)
	#g++ $(SRCDIR)/embedding.cc -o $(BINDIR)/embedding

bin/rnnContextRule: src/rnnContextRule.cc src/embedding.h
	mkdir -p bin
	g++ -std=c++11 $(CFLAGS) $(LIBS) $(INCS) $(SRCDIR)/rnnContextRule.cc -o $(BINDIR)/rnnContextRule $(FINAL)

bin/pro: src/pro.cc src/utils.h src/kbest_hypothesis.h src/pair_sampler.h
	mkdir -p $(BINDIR)
	g++ -std=c++11 $(CFLAGS) $(LIBS) $(INCS) $(SRCDIR)/pro.cc -o $(BINDIR)/pro $(FINAL)

bin/rerank: src/rerank.cc src/utils.h src/kbest_hypothesis.h src/pair_sampler.h
	mkdir -p $(BINDIR)
	g++ -std=c++11 $(CFLAGS) $(LIBS) $(INCS) $(SRCDIR)/rerank.cc -o $(BINDIR)/rerank $(FINAL)

clean:
	rm -rf $(BINDIR)/*
