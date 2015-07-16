#!/usr/bin/env python

# UNKS a corpus

import sys
import argparse
import codecs

PARSER = argparse.ArgumentParser(description="UNKs a corpus")
PARSER.add_argument("-i", "--input", type=str, dest="input", help="File to be unk-ed")
PARSER.add_argument("-p", "--prune", type=int, default=5, dest="prune", help="Prune at (freq)")
PARSER.add_argument("-o", "--output", type=str, dest="output", help="Output UNKed Corpus")
PARSER.add_argument("-v", "--vocab", type=str, dest="vocab", help="Vocab output")
PARSER.add_argument("-u", "--unk_token", type=str, default="<unk>", dest="unk", help="Vocab output")
opts = PARSER.parse_args()

if opts.input is None or opts.output is None or opts.vocab is None:
    PARSER.print_help()
    sys.exit(1)

UNK_TOKEN = opts.unk

corpus = codecs.open(opts.input, encoding="utf8")
outputCorpus = codecs.open(opts.output, "w+", encoding = "utf8")
outputVocab = codecs.open(opts.vocab, "w+", encoding = "utf8")

vocab = {}

for line in corpus:
    line = line.strip().split()
    for word in line:
        if word not in vocab:
            vocab[word] = 0
        vocab[word] += 1

# Prune the vocab
vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
prunedVocab = [x[0] for x in vocab if x[1] > opts.prune]
oovCands = set([x[0] for x in vocab if x[1] <= opts.prune])

corpus.seek(0)

for line in corpus:
    unkedLine = " ".join([word if word not in oovCands else UNK_TOKEN for word in line.strip().split()])
    outputCorpus.write(unkedLine + "\n")

for word in prunedVocab:
    outputVocab.write(word + "\n")

outputVocab.write(UNK_TOKEN)

outputVocab.close()
outputCorpus.close()
corpus.close()

