.PHONY: clean
all: bin/embedding bin/pro

bin/embedding: src/embedding.cc
	mkdir -p bin
	g++ src/embedding.cc -o bin/embedding

CNN_DIR = /Users/austinma/git/cnn/
bin/pro: src/pro.cc src/utils.h
	mkdir -p bin
	g++ -std=c++11 -L$(CNN_DIR)/build/cnn/ -I$(CNN_DIR) src/pro.cc -o bin/pro -lcnn

clean:
	rm -rf bin/embedding bin/pro
