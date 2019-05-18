import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import make_classification
import pandas


df = pandas.read_csv('iris_csv.csv')
U = df.values


X = np.ndarray(shape=(150,4), dtype=float) # , order='F'

for i in range(150):
	X[i] = U[i][:-1]

y = np.ndarray(shape=(150), dtype=int) # , order='F'

for i in range(150):
	if U[i][4]== 'Iris-setosa':
		y[i] = 1
	if U[i][4]== 'Iris-versicolor':
		y[i] = 2
	if U[i][4]== 'Iris-virginica':
		y[i] = 3


#####   		Task 1 
STNO = 14503897
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
	test_size=0.20, random_state = STNO)


#####   		Task 2 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm

C = [10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1, 10]
CS = ['10e-7', '10e-6', '10e-5', '10e-4', '10e-3', '10e-2', '10e-1', '1', '10']


AC_train = []
AC_test = []
for i in range(len(C)):
	clf = svm.LinearSVC(C = C[i]) 
	clf.fit(X_train, y_train)
	pred_train = clf.predict(X_train)
	pred_test = clf.predict(X_test)
	AC_train.append(accuracy_score(y_train, pred_train)*100)
	AC_test.append(accuracy_score(y_test, pred_test)*100)

k = np.argmax(AC_test)

fig, ax = plt.subplots()

plt.plot(CS,AC_train, 'ro-',CS, AC_test,  'bo-')
plt.grid()
plt.xlabel('Penalty parameter, C')
plt.ylabel('Accuracy')
plt.legend(['train', 'test'])
plt.title('LinearSVC, best C = '+str(C[k]))
plt.show()

#####   		Task 3
from sklearn import tree


D = list(range(1, 11))  # 1:10
AC_train = []
AC_test = []
for i in range(len(D)):
	clf = tree.DecisionTreeClassifier(max_depth = D[i])
	clf.fit(X_train, y_train)
	pred_train = clf.predict(X_train)
	pred_test = clf.predict(X_test)
	AC_train.append(accuracy_score(y_train, pred_train)*100)
	AC_test.append(accuracy_score(y_test, pred_test)*100)

Opt_depth = D[np.argmax(AC_test)]
AC_train_D =AC_train
AC_test_D = AC_test

F = list(range(1, 5))  # 1:10
AC_train = []
AC_test = []
for i in range(len(F)):
	clf = tree.DecisionTreeClassifier(max_depth = Opt_depth, max_features = F[i])
	clf.fit(X_train, y_train)
	pred_train = clf.predict(X_train)
	pred_test = clf.predict(X_test)
	AC_train.append(accuracy_score(y_train, pred_train)*100)
	AC_test.append(accuracy_score(y_test, pred_test)*100)

Opt_features = F[np.argmax(AC_test)]

clf = tree.DecisionTreeClassifier(max_depth = Opt_depth, 
								max_features = Opt_features)
clf.fit(X_train, y_train)

importances = clf.feature_importances_

f, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.plot(D, AC_train_D, 'ro-', D, AC_test_D,  'bo-')
ax1.set_ylim(0, 100)
ax1.grid()
ax1.set_title('DecisionTreeClassifier')
ax1.set_xlabel('max_depth')
ax1.set_ylabel('Accuracy')
ax1.legend(['train', 'test'])

ax2.plot(F, AC_train, 'ro-', F, AC_test,  'bo-')
ax2.set_ylim(0, 100)
ax2.grid()
ax2.set_title('max_depth = '+str(Opt_depth))
ax2.set_xlabel('max_features')
ax2.set_ylabel('Accuracy')
ax2.legend(['train', 'test'])

ax3.plot(importances, range(1, 5), 'bo-')
ax3.set_xlim(0, 1.1*np.max(importances))
ax3.grid()
ax3.set_title('max_depth = '+str(Opt_depth)+', max_features = ' + str(Opt_features))
ax3.set_xlabel('Feature importance')
ax3.set_ylabel('Feature')

plt.show()


#####   		Task 3
from sklearn.ensemble import RandomForestClassifier
F = list(range(1, 5))  # 1:10
AC_train = []
AC_test = []
for i in range(len(F)):
	clf = RandomForestClassifier( max_features = F[i])
	clf.fit(X_train, y_train)
	pred_train = clf.predict(X_train)
	pred_test = clf.predict(X_test)
	AC_train.append(accuracy_score(y_train, pred_train)*100)
	AC_test.append(accuracy_score(y_test, pred_test)*100)

Opt_features = F[np.argmax(AC_test)]

clf = RandomForestClassifier(max_features = Opt_features)
clf.fit(X_train, y_train)

importances = clf.feature_importances_

f, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(F, AC_train, 'ro-', F, AC_test,  'bo-')
ax1.set_ylim(0, 100)
ax1.grid()
ax1.set_title('RandomForestClassifier')
ax1.set_xlabel('max_features')
ax1.set_ylabel('Accuracy')
ax1.legend(['train', 'test'])

ax2.plot(importances, range(1, 5), 'bo-')
ax2.set_xlim(0, 1.1*np.max(importances))
ax2.grid()
ax2.set_title('max_features = ' + str(Opt_features))
ax2.set_xlabel('Feature importance')
ax2.set_ylabel('Feature')

plt.show()

#####   		Task 4

from sklearn.ensemble import GradientBoostingClassifier
D = list(range(1, 10))  # 1:10
AC_train = []
AC_test = []
for i in range(len(D)):
	clf = GradientBoostingClassifier( max_depth = D[i])
	clf.fit(X_train, y_train)
	pred_train = clf.predict(X_train)
	pred_test = clf.predict(X_test)
	AC_train.append(accuracy_score(y_train, pred_train)*100)
	AC_test.append(accuracy_score(y_test, pred_test)*100)

Opt_depth = D[np.argmax(AC_test)]

clf = GradientBoostingClassifier(max_depth = Opt_depth)
clf.fit(X_train, y_train)

importances = clf.feature_importances_

f, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(D, AC_train, 'ro-', D, AC_test,  'bo-')
ax1.set_ylim(0, 110)
ax1.grid()
ax1.set_title('GradientBoostingClassifier')
ax1.set_xlabel('max_features')
ax1.set_ylabel('Accuracy')
ax1.legend(['train', 'test'])

ax2.plot(importances, range(1, 5), 'bo-')
ax2.set_xlim(0, 1.1*np.max(importances))
ax2.grid()
ax2.set_title('max_depth = ' + str(Opt_depth))
ax2.set_xlabel('Feature importance')
ax2.set_ylabel('Feature')

plt.show()
