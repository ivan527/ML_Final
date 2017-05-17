from sys import *

import numpy as np
import csv
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

MAX_NUM_FEATURES  = 20

# Read in features, normalize and encode them, and put them into array
def read_data(input_file):
    X = []
    Y = []
    depression_no = 0
    with open(input_file, 'r') as data:
        reader = csv.reader(data)
        for row in reader:
            sample = []
            age = float(row[26]) # c1
            sex = int(row[28]) #c2
            race = int(row[2989]) #c3 - c7
            bioparents = int(row[33]) # c8 - did not care for unknown, if it was unknown treated it as biological
            parent_divorce = int(row[40]) # c9 - got rid of unknown
            parent_death = int(row[47]) # c10 - got rid of unknown
            marital = int(row[49]) #c11 - c16
            education = float(row[59]) #c17
            employment = int(row[88]) #c18
            income = float(row[102]) #c19
            health = float(row[124]) #c20
            emotional_activities = float(row[129]) #c21
            recent_death = int(row[156]) #c22
            inches = float(row[168]) #c23
            feet = float(row[170]) #c23
            weight = float(row[170]) #c24
            drink_beer = int(row[186]) #c25 - c35
            drink_liquor = int(row[206]) #c36 - c46
            alcoholic_father = int(row[429]) #c47
            alcoholic_mother = int(row[430]) #c48
            tobacco = float(row[472]) #c49
            sedatives = int(row[613]) # c50 - grouped unknown with no
            tranquilizers = int(row[614]) #c51 - same
            opioids = int(row[615]) #c52 - same
            amphetamines = int(row[616]) #c53 - same
            cannabis = int(row[617]) #c54 - same
            cocaine = int(row[618]) #c55 - same
            hallucinogens = int(row[619]) #c56 - same
            inhalants = int(row[620]) #c57 - same
            heroin = int(row[621]) #c58 - same
            other_drugs = int(row[622]) #c59 - same
            attempted_suicide = int(row[1897]) #c60 - grouped unknown and NA with no
            thought_suicide = int(row[1898]) #c61 - same
            thought_death = int(row[1899]) #c62 - same
            want_alone = int(row[1905]) #c63 - same
            lost_appetite = int(row[1982]) #c64 - same
            abnormal_appetite = int(row[1983]) #c65 - same
            sleeping_issues = int(row[1984]) #c66 - same
            sleep_alot = int(row[1985]) #c67 - same
            often_tired = int(row[1986]) #c68 -  same
            trouble_concentrate = int(row[1987]) #c69 - same
            trouble_decide = int(row[1988]) #c70 - same
            not_good = int(row[1989]) #c71 - same
            felt_down = int(row[1990]) #c72 - same
            never_improve = int(row[1991]) #c73 - same
            felt_hopeless = int(row[1992]) #c74 - same
            wish_better = int(row[1994]) #c75 - same
            fear_social = int(row[2189]) #c76 -discard unknown
            fear_embarassement = int(row[2190]) #c77 - same
            fear_speechless = int(row[2191]) #c78 - same
            fear_talking = int(row[2192]) #c79 - grouped unknown and NA with no
            depression = int(row[2865]) #label

            sample.append((age - 18) / 72)
            sample.append(sex % 2)
            for i in range(race - 1):
                sample.append(0)
            sample.append(1)
            for i in range(5 - race):
                sample.append(0)
            if bioparents == 9:
                continue
            sample.append(bioparents % 2)
            if parent_divorce == 1:
                sample.append(1)
            elif parent_divorce == 2:
                sample.append(0)
            else:
                continue
            if parent_death == 1:
                sample.append(1)
            elif parent_death == 2:
                sample.append(0)
            else:
                continue
            for i in range(marital - 1):
                sample.append(0)
            sample.append(1)
            for i in range(6 - marital):
                sample.append(0)
            sample.append((education - 1) / 13)
            sample.append(1) if employment == 1 else sample.append(0)
            sample.append((income - 1) / 20)
            if health == 9:
                continue
            sample.append((health - 1) / 4)
            if emotional_activities == 9:
                continue
            sample.append((emotional_activities - 1) / 4)
            if recent_death == 9:
                continue
            sample.append(recent_death % 2)
            sample.append((feet * 12 + inches - 48) / 48) #height
            sample.append((weight - 62) / 438)
            if drink_beer == 99:
                continue
            if drink_beer > 0:
                for i in range(drink_beer - 1):
                    sample.append(0)
                sample.append(1)
                for i in range(11 - drink_beer):
                    sample.append(0)
            else:
                for i in range(10):
                    sample.append(0)
                sample.append(1)
            if drink_liquor == 99:
                continue
            if drink_liquor > 0:
                for i in range(drink_liquor - 1):
                    sample.append(0)
                sample.append(1)
                for i in range(11 - drink_liquor):
                    sample.append(0)
            else:
                for i in range(10):
                    sample.append(0)
                sample.append(1)
            if alcoholic_father == 9:
                continue
            sample.append(alcoholic_father % 2)
            if alcoholic_mother == 9:
                continue
            sample.append(alcoholic_mother % 2)
            sample.append((tobacco - 1) / 3)
            sample.append(1) if sedatives == 1 else sample.append(0)
            sample.append(1) if tranquilizers == 1 else sample.append(0)
            sample.append(1) if opioids == 1 else sample.append(0)
            sample.append(1) if amphetamines == 1 else sample.append(0)
            sample.append(1) if cannabis == 1 else sample.append(0)
            sample.append(1) if cocaine == 1 else sample.append(0)
            sample.append(1) if hallucinogens == 1 else sample.append(0)
            sample.append(1) if inhalants == 1 else sample.append(0)
            sample.append(1) if heroin == 1 else sample.append(0)
            sample.append(1) if other_drugs == 1 else sample.append(0)
            sample.append(1) if attempted_suicide == 1 else sample.append(0)
            sample.append(1) if thought_suicide == 1 else sample.append(0)
            sample.append(1) if thought_death == 1 else sample.append(0)
            sample.append(1) if want_alone == 1 else sample.append(0)
            sample.append(1) if lost_appetite == 1 else sample.append(0)
            sample.append(1) if abnormal_appetite == 1 else sample.append(0)
            sample.append(1) if sleeping_issues == 1 else sample.append(0)
            sample.append(1) if sleep_alot == 1 else sample.append(0)
            sample.append(1) if often_tired == 1 else sample.append(0)
            sample.append(1) if trouble_concentrate == 1 else sample.append(0)
            sample.append(1) if trouble_decide == 1 else sample.append(0)
            sample.append(1) if not_good == 1 else sample.append(0)
            sample.append(1) if felt_down == 1 else sample.append(0)
            sample.append(1) if never_improve == 1 else sample.append(0)
            sample.append(1) if felt_hopeless == 1 else sample.append(0)
            sample.append(1) if wish_better == 1 else sample.append(0)
            if fear_social == 9:
                continue
            sample.append(fear_social % 2)
            if fear_embarassement == 9:
                continue
            sample.append(fear_embarassement % 2)
            if fear_speechless == 9:
                continue
            sample.append(fear_speechless % 2)
            sample.append(1) if fear_talking == 1 else sample.append(0)
            X.append(sample)
            Y.append(depression)
            if depression == 0:
                depression_no += 1
    # print(depression_no)
    # print(len(Y))
    return np.array(X), np.matrix(Y).T

class ForwardFeatureSelector:

	def __init__(self):
		self._selected_features = []
		self._selected_features_index = []
		self._classifier = None


	def fit(self, X, Y):
		# Iterate through all features, training NB on each feature. Pick feature with lowest training error with Leave One Out cross validation
		selected_features = self._selected_features
		selected_features_index = self._selected_features_index
		N = np.shape(X)[0]

		index = list(range(np.shape(X)[1]))
		index = np.array(index)
		unselected_features = np.vstack((index, X))
		max_index = -1

		


		for j in range(MAX_NUM_FEATURES):
			train_accuracies = []
			mean_score = []

			# Delete selected feature from unselected features
			if max_index != -1:
				print("selected feature at index: ", max_index)
				np.delete(unselected_features, max_index, 1)
			
			# Train and get scores of selected features + 1 unselected feature
			feats_left = len(np.array(unselected_features)[0])
			for feat in range(feats_left):
				# Concatenate feature to current selected features
				training_features = np.copy(selected_features)
				if len(selected_features) == 0:
					training_features = np.copy(unselected_features[:,feat])
					training_features = np.matrix([training_features]).T
				else:
					np.concatenate((training_features, unselected_features[:,feat]), axis=1)

				# Calculate mean score of LOO cross validation using Gaussian Naive Bayes
				clf = GaussianNB()
				print(np.shape(training_features[1:]), np.shape(Y.reshape(N,)))
				m_score = np.mean(cross_val_score(clf, training_features[1:], Y.reshape(N,), cv=10))
				mean_score.append(m_score)

			# Find feature with highest score and select that feature
			max_index = mean_score.index(max(mean_score))
			selected_features_index.append(unselected_features[0,max_index])
			if len(selected_features) == 0:
				selected_features = np.copy( unselected_features[:,max_index])
			else:
				np.concatenate((selected_features, unselected_features[:,max_index]), axis=1)

		# Save classifer with selected features
		clf = GaussianNB()
		clf.fit(selected_features[1:], Y)
		self._classifier = clf
		self._selected_features = selected_features
		self._selected_features_index = selected_features_index
		return clf


	def accuracy(self, X, Y):
		clf = self._classifier
		return clf.score(X, Y)




if __name__ == '__main__':
	if (len(argv) != 3):
		print("Usage: python NB.py <train file> <test_file>")

	train_X, train_Y = read_data(argv[1])

	clf = ForwardFeatureSelector()
	clf.fit(train_X, train_Y)

	test_X, test_Y = read_data(argv[2])

	print(clf.accuracy(test_X, test_Y))
	print(clf._selected_features_index)
