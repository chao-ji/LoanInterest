import pickle
from dateutil.parser import parse
from collections import Counter
import numpy as np
data = pickle.load(open('data_train.pickle'))
missing = [0 for i in range(32)]

# I. Prepare data

for i in range(len(data)):
	csv = data[i]
	# Convert numerical and ordinal text data into float format
	for j in range(0, 1) + range(3, 6) + range(10, 11) + range(12, 13) + range(14, 15) + range(20, 31):
		if len(csv[j]) > 0:
			# percentage
			if j == 0 or j == 20 or j == 29:
				csv[j] = str(float(csv[j]) / 100.0)
			# number of months back w.r.t. 1-Aug-2016
			elif j == 14 or j == 22:
				delta = parse('1-Aug-2016') - parse(csv[j])
				csv[j] = delta.days / 30.0
			csv[j] = float(csv[j])
	data[i] = csv

# II. Impute missing values
# sort data w.r.t to interest rate, then split data into 20 ranges, and computed conditional/local mean or mode in each range as imputed values
data = sorted(data, key = lambda x : x[0])

nbins = 20 
binsize = len(data) / nbins
delim = range(0, len(data), binsize) + [len(data)]

num_index = [12, 24, 25, 29] # [12, 24, 25, 29] 
cat_index = [7, 8, 10, 11]

# Compute global mean of numerical features
num_mean = dict()
for j in num_index:
	values = []
	X = map(lambda x:x[j], data)
	for k in range(len(X)):
		if data[k][j] != '':
			values.append(data[k][j])
	mean = sum(values) / float(len(values))
	num_mean[j] = mean
# Compute global mode of categorical/ordinal features
cat_mode = dict()
for j in cat_index:
	values = []
	X = map(lambda x:x[j], data)
	for k in range(len(X)):
		if data[k][j] != '':
			values.append(data[k][j])
	mode = Counter(values)
	mode = mode.most_common(1)[0][0]
	cat_mode[j] = mode

# Compute local/conditional mean or mode of numerical or categorical/ordinal features
for i in range(len(delim) - 1):
	start = delim[i]
	end = delim[i + 1]

	# numerical feature imputation: mean
	for j in num_index:
		values = []
		for k in range(start, end):
			if data[k][j] != '':
				values.append(data[k][j])
		mean = sum(values) / float(len(values))

		for k in range(start, end):
			if data[k][j] == '':

				# imputed value = conditional mean if local sample size is large enough, else global mean
				data[k][j] = mean if len(values) >= 20000 else num_mean[j]
	# categorical or ordinal feature imputation: mode
	for j in cat_index:
		values = []
		for k in range(start, end):
			if data[k][j] != '':
				values.append(data[k][j])
		mode = Counter(values)
		mode = mode.most_common(1)[0][0]

		for k in range(start, end):
			if data[k][j] == '':

				if j == 7 or j == 8:
					data[k][j] = mode if len(values) >= 100 else cat_mode[j]
				else:
					data[k][j] = mode if len(values) >= 20000 else cat_mode[j]

# III. Construct training set
X = []
y = []

real_feature_index = [3, 4, 5, 12, 20, 22, 28, 29] #[3, 4, 5, 12, 20, 22, 28, 29]
one_hot_index = [6, 11, 13, 31]

one_hot = {6 : {	'36' : [1.0, 0.0],
					'60' : [0.0, 1.0]},
			11 : {	'RENT': [1.0, 0.0, 0.0],
					'OWN' : [0.0, 1.0, 0.0],
					'MORTGAGE' : [0.0, 0.0, 1.0]},
			13 : {	'VERIFIED - income' : [1.0, 0.0, 0.0],
					'VERIFIED - income source' : [0.0, 1.0, 0.0],
					'not verified' : [0.0, 0.0, 1.0]},
			31 : {	'w' : [1.0, 0.0],
					'f' : [0.0, 1.0]}}

subgrade = {'A1' : 1.0,	'A2' : 2.0, 'A3' : 3.0, 'A4' : 4.0, 'A5': 5.0,
			'B1' : 6.0, 'B2' : 7.0, 'B3' : 8.0, 'B4' : 9.0, 'B5': '10.0',
			'C1' : 11.0, 'C2' : 12.0, 'C3' : 13.0, 'C4' : 14.0, 'C5': 15.0,
			'D1' : 16.0, 'D2' : 17.0, 'D3' : 18.0, 'D4' : 19.0, 'D5': 20.0,
			'E1' : 21.0, 'E2' : 22.0, 'E3' : 23.0, 'E4' : 24.0, 'E5': 25.0,
			'F1' : 26.0, 'F2' : 27.0, 'F3' : 28.0, 'F4' : 29.0, 'F5': 30.0,
			'G1' : 31.0, 'G2' : 32.0, 'G3' : 33.0, 'G4' : 34.0, 'G5': 35.0}

transform_index = [21, 23, 24, 25, 26, 27, 30] #[21, 23, 24, 25, 26, 27, 30]

for i in range(len(data)):
	y.append(data[i][0])
	x = []

	# 8 
	for j in real_feature_index:
		x.append(data[i][j])

	# 7
	# discretize counts (e.g. 'Number of ...')
	for j in transform_index:
		val = round(data[i][j])
		if j == 21:
			valp = val if val < 3.0 else 3.0
		elif j == 23:
			valp = val if val < 6.0 else 6.0
		elif j == 24:
			valp = val if val == 0.0 else 1.0 / val	# 0.0 means no delinquency (no penalty); 1.0 means very recent deliquency incidence (maximum penalty); > 1.0 means not recent delinquency (less penalty)
		elif j == 25:
			valp = val if val < 1.0 else 1.0
		elif j == 26:
			valp = val if val < 4.0 else 4.0
		elif j == 27:
			valp = val if val < 2.0 else 2.0
		elif j == 30:
			valp = val if val < 20.0 else 20.0
		x.append(valp)

	# 2
	x.append(subgrade[data[i][8]])	# subgrade
	x.append(data[i][10])			# number of years of employment

	# 2 + 3 + 3 + 2
	for j in one_hot_index:
		x.extend(one_hot[j][data[i][j]])

	X.append(x)

X = np.array(X, dtype='float')
y = np.array(y, dtype='float')

index = np.arange(X.shape[0])
np.random.shuffle(index)
X = X[index, :]
y = y[index]

pickle.dump([X, y], open('train.pickle', 'w'))
