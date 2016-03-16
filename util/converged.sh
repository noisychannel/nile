#!/usr/bin/env bash

for exp in `seq 1 16`; do
  for r in `seq 1 10`; do 
    size="full"
    if [ $exp -lt 9 ]; then 
      size="medium"
    fi
    tail -n12 log5/${size}.${exp}.${r}.err \
      | grep -oh "EBLEU = [^)]*" \
      | awk '{print $3}' \
      | awk -v run="$r" 'BEGIN {c=-1} {c+=1;s[c]=$1;} END {for (x = 0; x <= c; x++) {if (s[x] != s[0]) {print run, " : Not converged"; exit;}}; print run, ": Coverged"}';
  done \
  | awk '{print $3}' | uniq | awk -v e="$exp" 'BEGIN{converged=1} {if ($1 == "Not") {converged=0}} END{if (converged) {print e" : Converged"} else {print e" : Not converged"}}';
done
