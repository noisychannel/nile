import sys
import argparse
from math import sqrt

parser = argparse.ArgumentParser()
parser.add_argument('w')
parser.add_argument('v')
args = parser.parse_args()

def read_vector(filename):
	r = {}
	with open(filename) as f:
		for line in f:
			name, value = line.split()
			value = float(value.strip())
			r[name] = value
	return r

w = read_vector(args.w)
v = read_vector(args.v)

s = 0.0
mw = sqrt(sum(val**2 for val in w.values()))
mv = sqrt(sum(val**2 for val in v.values()))
for k, wv in w.iteritems():
	if k in v:
		vv = v[k]
		s += wv * vv

print 'Dot product:', s
print '||w||:', mw
print '||v||:', mv
print 'Cosine similarity:', s / (mw * mv)
