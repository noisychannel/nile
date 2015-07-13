import sys
import argparse
from math import sqrt
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('kbest')
args = parser.parse_args()

feat_counts = defaultdict(int)
feat_sums = defaultdict(float)
feat_vars = defaultdict(float)

print >>sys.stderr, 'Computing means...'
f = open(args.kbest)
for line in f:
	parts = line.split('|||')
	features = parts[2].strip()
	features = features.split()
	features = [kvp.split('=') for kvp in features]
	features = {k:float(v) for k, v in features}

	for k, v in features.iteritems():
		feat_sums[k] += v
		feat_counts[k] += 1
f.close()

print >>sys.stderr, 'Normalizing...'
for k in feat_sums.keys():
	feat_sums[k] /= feat_counts[k]

print >>sys.stderr, 'Computing variances...'
f = open(args.kbest)
for line in f:
	parts = line.split('|||')
	features = parts[2].strip()
	features = features.split()
	features = [kvp.split('=') for kvp in features]
	features = {k:float(v) for k, v in features}

	for k, v in features.iteritems():
		feat_vars[k] += (v - feat_sums[k]) ** 2
f.close()

print >>sys.stderr, 'Normalizing...'
for k in feat_vars.keys():
	feat_vars[k] = sqrt(feat_vars[k] / feat_counts[k])

print >>sys.stderr, 'Output...'
f = open(args.kbest)
for line in f:
	parts = line.split('|||')
	parts = [part.strip() for part in parts]
	features = parts[2].strip()
	features = features.split()
	features = [kvp.split('=') for kvp in features]
	features = {k:float(v) for k, v in features}
	features = {k: (v - feat_sums[k]) / feat_vars[k] if abs(feat_vars[k]) > 1.0e-3 else feat_sums[k] for k, v in features.iteritems()}
	parts[2] = ' '.join('%s=%f' % (k, v) for k, v in features.iteritems())
	print ' ||| '.join(parts)
f.close()

