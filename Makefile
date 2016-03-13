CC=g++
EIGEN=./.eigen
CNN_DIR=./.cnn
CNN_BUILD_DIR=$(CNN_DIR)/build
INCS=-I$(CNN_DIR) -I$(CNN_BUILD_DIR) -I$(EIGEN)
LIBS=-L$(CNN_BUILD_DIR)/cnn/
FINAL=-lcnn -lboost_regex -lboost_serialization -lboost_program_options
CFLAGS=-std=c++11 -Ofast -march=native -pipe
#CFLAGS=-std=c++11 -O0 -g -DDEBUG -pipe -gdwarf-3
BINDIR=bin
OBJDIR=obj
SRCDIR=src

.PHONY: clean
all: make_dirs $(BINDIR)/rerank $(BINDIR)/train

make_dirs:
	mkdir -p $(OBJDIR)
	mkdir -p $(BINDIR)

clean:
	rm -rf $(BINDIR)/*
	rm -rf $(OBJDIR)/*

include $(wildcard $(OBJDIR)/*.d)

$(OBJDIR)/%.o: $(SRCDIR)/%.cc
	$(CC) $(CFLAGS) $(INCS) -c $< -o $@
	$(CC) -MM -MP -MT "$@" $(CFLAGS) $(INCS) $< > $(OBJDIR)/$*.d
	
$(BINDIR)/train: $(addprefix $(OBJDIR)/, train.o kbestlist.o utils.o kbest_hypothesis.o reranker.o feature_extractor.o dataview.o gaurav.o context.o)
	g++ $(LIBS) $^ -o $@ $(FINAL)

$(BINDIR)/rerank: $(addprefix $(OBJDIR)/, rerank.o kbestlist.o utils.o kbest_hypothesis.o reranker.o feature_extractor.o dataview.o gaurav.o context.o)
	g++ $(LIBS) $^ -o $@ $(FINAL)

$(BINDIR)/sandbox: $(addprefix $(OBJDIR)/, sandbox.o kbest_hypothesis.o utils.o)
	g++ $(LIBS) $^ -o $@ $(FINAL)
