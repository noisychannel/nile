#!/usr/bin/env python

# UNKS a corpus

import sys
import argparse
import codecs

parser = argparse.ArgumentParser(description="UNKs a corpus")
parser.add_argument("-i", "--input", type=str, dest="input", help="File to be unk-ed")
parser.add_argument("-p", "--prune", type=int, default=5, dest="prune", help="Prune at (freq)")
parser.add_argument("-o", "--output", type=str, dest="output", help="Output UNKed Corpus")
parser.add_argument("-v", "--vocab", type=str, dest="vocab", help="Vocab output")
parser.add_argument("-u", "--unk_token", type=str, default="<unk>", dest="unk", help="Vocab output")
opts = parser.parse_args()

if opts.input is None or opts.output is None or opts.vocab is None:
    parser.print_help()
    sys.exit(1)

unk_token = opts.unk

corpus = codecs.open(opts.input, encoding="utf8")
output_corpus = codecs.open(opts.output, "w+", encoding = "utf8")
output_vocab = codecs.open(opts.vocab, "w+", encoding = "utf8")

vocab = {}

for line in corpus:
    line = line.strip().split()
    for word in line:
        if word not in vocab:
            vocab[word] = 0
        vocab[word] += 1

# Prune the vocab
vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
pruned_vocab = [x[0] for x in vocab if x[1] >= opts.prune]
oov_cands = set([x[0] for x in vocab if x[1] < opts.prune])

corpus.seek(0)

for line in corpus:
    unkedLine = " ".join([word if word not in oov_cands else unk_token for word in line.strip().split()])
    output_corpus.write(unkedLine + "\n")

for word in pruned_vocab:
    output_vocab.write(word + "\n")

output_vocab.write(unk_token)

output_vocab.close()
output_corpus.close()
corpus.close()

