import sys
import argparse
import cPickle as pickle
from math import sqrt
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('kbest')
parser.add_argument('--input_file', '-i', type=str, required=False, default=None, help='File with stored statistics to use, instead of recomputing')
parser.add_argument('--output_file', '-o', type=str, required=False, default=None, help='File in which to store statistics')
args = parser.parse_args()

if args.output_file:
	output_file = open(args.output_file, 'w')

feat_sums = defaultdict(float)
feat_vars = defaultdict(float)

if not args.input_file:
	feat_counts = defaultdict(int)
	print >>sys.stderr, 'Computing means...'
	f = open(args.kbest)
	n = 0
	for line in f:
		n += 1
		if n % 50000 == 0:
			sys.stderr.write('.')
		parts = line.split('|||')
		features = parts[2].strip()
		features = features.split()
		features = [kvp.split('=') for kvp in features]
		features = {k:float(v) for k, v in features}

		for k, v in features.iteritems():
			feat_sums[k] += v
			feat_counts[k] += 1
	f.close()
	print >>sys.stderr

	print >>sys.stderr, 'Normalizing...'
	for k in feat_sums.keys():
		feat_sums[k] /= feat_counts[k]

	print >>sys.stderr, 'Computing variances...'
	f = open(args.kbest)
	n = 0
	for line in f:
		n += 1
		if n % 50000 == 0:
			sys.stderr.write('.')
		parts = line.split('|||')
		features = parts[2].strip()
		features = features.split()
		features = [kvp.split('=') for kvp in features]
		features = {k:float(v) for k, v in features}

		for k, v in features.iteritems():
			feat_vars[k] += (v - feat_sums[k]) ** 2
	f.close()
	print >>sys.stderr

	print >>sys.stderr, 'Normalizing...'
	for k in feat_vars.keys():
		feat_vars[k] = sqrt(feat_vars[k] / feat_counts[k])
else:
	print >>sys.stderr, 'Loading...'
	input_file = open(args.input_file)
	feat_sums, feat_vars = pickle.load(input_file)
	input_file.close()

if args.output_file:
	pickle.dump((feat_sums, feat_vars), output_file)
	output_file.close()


print >>sys.stderr, 'Output...'
f = open(args.kbest)
n = 0
for line in f:
	n += 1
	if n % 50000 == 0:
		sys.stderr.write('.')
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
print >>sys.stderr

