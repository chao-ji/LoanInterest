import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn import linear_model
#from sklearn import svm
from sklearn.decomposition import PCA

model = BaggingRegressor(linear_model.Ridge(alpha = 0.6), n_estimators = 50)

(X, y) = pickle.load(open('train.pickle'))

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

X = X[:, 12, np.newaxis]

#pca = PCA(n_components=0.99)
#pca.fit(X)
#X = pca.transform(X)

#print X.shape[0]
#print X.shape[1]

n_sample = X.shape[0]
kfcv = 10 

TR = []
TS = []
kf = KFold(n_sample, n_folds = kfcv)

for train, test in kf:
	TR.append(train)
	TS.append(test)

A = []
B = []

for k in range(kfcv):
	print k
	X_train = X[TR[k], :]
	y_train = y[TR[k]]
	X_test = X[TS[k], :]
	y_test = y[TS[k]]

	model.fit(X_train, y_train)
	y_predict = model.predict(X_test)

#	plt.subplot(2, 10, k + 1)
#	plt.scatter(y_predict, y_test)
#	plt.xlabel('y_predict')	
#	plt.ylabel('y_true')
#	plt.title('Fold = %d' % (k + 1))

	A.extend(list(y_predict))
	B.extend(list(y_test))

	mse = mean_squared_error(y_predict, y_test)
	print 'mse = %f' % mse

mse = mean_squared_error(A, B)

#model = BaggingRegressor(linear_model.Ridge(alpha = 0.6), n_estimators = 50)
#model.fit(X, y)
#pickle.dump((model, scaler), open('ridge.pickle', 'w'))

print mse
