#!/usr/bin/env bash

devN=jsalt/data/dense/dev1.output.1.best300
trainN=jsalt/data/dense/dev2.output.1.best300
testN=jsalt/data/dense/test.output.1.best300

if [ -e ${devN} ]; then
  # Convert to moses format
  cat ${devN} | python util/conv_moses.py > ${devN}.conv
  # Normalize
  python util/normalize.py ${devN}.conv > ${devN}.conv.norm
  # Get sentence BLEU
  python util/compute_bleu.py ${devN}.conv.norm > ${devN}.conv.norm.bleu
fi

if [ -e ${testN} ]; then
  # Convert to moses format
  cat ${testN} | python util/conv_moses.py > ${testN}.conv
  # Normalize
  python util/normalize.py ${testN}.conv > ${testN}.conv.norm
  # Get sentence BLEU
  python util/compute_bleu.py ${testN}.conv.norm > ${testN}.conv.norm.bleu
fi

if [ -e ${trainN} ]; then
  # Convert to moses format
  cat ${trainN} | python util/conv_moses.py > ${trainN}.conv
  # Normalize
  python util/normalize.py ${trainN}.conv > ${trainN}.conv.norm
  # Get sentence BLEU
  python util/compute_bleu.py ${trainN}.conv.norm > ${trainN}.conv.norm.bleu
fi

# Get sentence BLEU
