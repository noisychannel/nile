import sys

for line in sys.stdin:
	parts = [part.strip() for part in line.strip().split('|||')]
	moses_features = parts[2]
	feat_name = None
	index = 0
	features = []
	for token in [token.strip() for token in moses_features.split()]:
		if token.endswith('='):
			feat_name = token[:-1]
			index = 0
		else:
			assert feat_name != None
			features.append('%s_%d=%s' % (feat_name, index, token))
			index += 1
	parts[2] = ' '.join(features)
	print ' ||| '.join(parts)
