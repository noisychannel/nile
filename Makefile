bin/embedding: src/embedding.cc
	mkdir -p bin
	g++ src/embedding.cc -o bin/embedding
