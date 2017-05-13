from sys import *
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import GaussianNB

# Read in features, normalize and encode them, and put them into array
def read_data(input_file):

	return X, Y

class ForwardFeatureSelector:

	def __init__(self):
		selected_features = []
		
	
	def fit(self, X, Y):
		# Iterate through all features, training NB on each feature. Pick feature with lowest training error with Leave One Out cross validation

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
