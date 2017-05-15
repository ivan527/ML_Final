from sys import *
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

MAX_NUM_FEATURES  = 20

# Read in features, normalize and encode them, and put them into array
def read_data(input_file):

	return X, Y

class ForwardFeatureSelector:

	def __init__(self):
		self._selected_features = None
		self._selected_features_index = None
		self._classifier = None
		
	
	def fit(self, X, Y):
		# Iterate through all features, training NB on each feature. Pick feature with lowest training error with Leave One Out cross validation
		selected_features = self._selected_features
		selected_features_index = self._selected_features_index
		N = len(X)

		index = list(range(len(X[:,0])))
		X.insert(0, index)


		for i in range(MAX_NUM_FEATURES):
			train_accuracies = []
			mean_score = []
			for feat in index:
				# Concatenate feature to current selected features
				if selected_features == None:
					selected_features = X[:,feat]
				else:
					np.concatenate((selected_features, X[:,feat]), axis=1)

				# Calculate mean score of LOO cross validation using Gaussian Naive Bayes
				clf = GaussianNB()
				mean_score += [numpy.mean(cross_val_score(clf, X[1:], Y, cv=N))]

			# Find feature with highest score and select that feature
			for feat in index:
		
		# Save classifer with selected features
		clf = GaussianNB()
		clf.fit(selected_features, Y)
		self._classifier = clf
		return clf




	def predict(self, X):

	def accuracy(self, X, Y):
		predicted = predict(X)
		

		

if __name__ == '__main__':
	if (len(argv) != 3):
		print("Usage: python NB.py <train file> <test_file>")
	
	train_X, train_Y = read_data(argv[1])

	clf = ForwardFeatureSelector()
	clf.fit(train_X, train_Y)
	
	test_X, test_Y = read_data(argv[2])

	clf.accuracy(test_X, test_Y)
