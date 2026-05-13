from sklearn.svm import LinearSVC
from sklearn.metrics import *
from sklearn.cluster import KMeans
import warnings
import numpy as np


def node_clustering(X, y=None, n_clusters=None, seed=0, n_init=10):
	if n_clusters is None:
		if y is None:
			raise ValueError("n_clusters must be provided when y is None.")
		y = np.asarray(y)
		if y.ndim > 1:
			y = np.argmax(y, axis=1)
		n_clusters = np.unique(y).shape[0]

	kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init)
	return kmeans.fit_predict(X).astype(np.int64)

def multiclass_node_classification_eval(X, y, ratio=0.3, seed=0):
	state = np.random.get_state()
	np.random.seed(seed)
	y = np.array(y)
	warnings.filterwarnings("ignore")
	num_nodes = len(X)
	shuffle_indices = np.random.permutation(np.arange(num_nodes))
	train_idx = shuffle_indices[:int(ratio * num_nodes)]
	test_idx = shuffle_indices[int(ratio * num_nodes):]
	X_train = X[train_idx]
	y_train = y[train_idx]
	X_test = X[test_idx]
	y_test = y[test_idx]
	clf = LinearSVC()
	clf.fit(X_train, y_train)
	test_y_pred = clf.predict(X_test)
	test_macro_f1 = f1_score(y_test, test_y_pred, average = "macro")
	test_micro_f1 = f1_score(y_test, test_y_pred, average = "micro")
	np.random.set_state(state)
	return test_macro_f1, test_micro_f1
