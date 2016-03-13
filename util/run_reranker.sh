#!/usr/bin/env bash

devN=zh-data/dev.best300
trainN=zh-data/tune.best300
testN=zh-data/test.best300
devR=zh-data/dev.1.en
trainR=zh-data/tune.1.en
testR=zh-data/test.1.en

if [ -e ${devN} ]; then
  # Convert to moses format
  cat ${devN} | python util/conv_moses.py > ${devN}.conv
  # Normalize
  python util/normalize.py ${devN}.conv > ${devN}.conv.norm
  # Get sentence BLEU
  python util/compute_bleu.py ${devN}.conv.norm ${devR} > ${devN}.conv.norm.bleu
fi

if [ -e ${testN} ]; then
  # Convert to moses format
  cat ${testN} | python util/conv_moses.py > ${testN}.conv
  # Normalize
  python util/normalize.py ${testN}.conv > ${testN}.conv.norm
  # Get sentence BLEU
  python util/compute_bleu.py ${testN}.conv.norm ${testR} > ${testN}.conv.norm.bleu
fi

if [ -e ${trainN} ]; then
  # Convert to moses format
  cat ${trainN} | python util/conv_moses.py > ${trainN}.conv
  # Normalize
  python util/normalize.py ${trainN}.conv > ${trainN}.conv.norm
  # Get sentence BLEU
  python util/compute_bleu.py ${trainN}.conv.norm ${trainR} > ${trainN}.conv.norm.bleu
fi

# Get sentence BLEU
