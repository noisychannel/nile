#!/usr/bin/env python

import sys

nbest_file = open(sys.argv[1])
max_sents = int(sys.argv[2])
max_best = int(sys.argv[3])

sent_count = 0
sent_id = -1
best_count = 0
for hyp in nbest_file:
    if sent_count >= max_sents and best_count == max_best:
        break
    hyp_comp = hyp.split("|||")
    current_sent_id = hyp_comp[0]
    if current_sent_id != sent_id:
        sent_id = current_sent_id
        best_count = 0
        sent_count += 1
    if best_count < max_best:
        #print current_sent_id, best_count, max_best
        print hyp.strip()
        best_count += 1

nbest_file.close()
