CC=g++
#CNN_DIR = /Users/austinma/git/cnn/
CNN_DIR=/export/a04/gkumar/code/cnn/
EIGEN=/export/a04/gkumar/code/eigen/
CNN_BUILD_DIR=$(CNN_DIR)/build
INCS=-I$(CNN_DIR) -I$(CNN_BUILD_DIR) -I$(EIGEN)
LIBS=-L$(CNN_BUILD_DIR)/cnn/
FINAL=-lcnn
CFLAGS=
BINDIR=bin
SRCDIR=src

.PHONY: clean
all: $(BINDIR)/embedding $(BINDIR)/pro $(BINDIR)/rnnContextRule
#all: $(BINDIR)/embedding $(BINDIR)/pro

$(BINDIR)/embedding: $(SRCDIR)/embedding.cc
	mkdir -p $(BINDIR)
	g++ $(SRCDIR)/embedding.cc -o $(BINDIR)/embedding

bin/rnnContextRule: src/rnnContextRule.cc
	mkdir -p bin
	g++ -std=c++11 $(CFLAGS) $(LIBS) $(INCS) $(SRCDIR)/rnnContextRule.cc -o $(BINDIR)/rnnContextRule $(FINAL)

bin/pro: src/pro.cc src/utils.h
	mkdir -p $(BINDIR)
	g++ -std=c++11 $(CFLAGS) $(LIBS) $(INCS) $(SRCDIR)/pro.cc -o $(BINDIR)/pro $(FINAL)

clean:
	rm -rf $(BINDIR)/*
