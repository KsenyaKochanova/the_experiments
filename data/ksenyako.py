__author__ = 'ksenyako'
import numpy as np
import scipy


def swap(index1, index2, iterable):
	for x in iterable:
		x[index1],x[index2]=x[index2],x[index1]

raw_data = open ('data.txt')
dataset = np.loadtxt(raw_data,delimiter=",")
matr = dataset[:,:]

swap(0,6,matr)

X_all = matr[:,1:]
y_all = matr [:,0]
X_train = matr [0:150,1:]
y_train = matr [0:150,0]
X_test = matr[150:194,1:]
y_test = matr [150:194,0]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_all = scaler.fit_transform(X_all)

from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB()
model.fit(X_train, y_train)
expected = y_test
predicted = model.predict(X_test)
print predicted

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2,
                               min_samples_leaf=1,
                               min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                               bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
                               warm_start=False, class_weight=None)
model.fit(X_train, y_train)
predicted = model.predict(X_test)
expected = y_test
print predicted


from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=10,
                                   subsample=1.0, min_samples_split=2, min_samples_leaf=1,
                                   min_weight_fraction_leaf=0.0, max_depth=1, init=None,
                                   random_state=None, max_features=None, verbose=0,
                                   max_leaf_nodes=None, warm_start=False)
model.fit(X_train, y_train)
expected = y_test
predicted = model.predict(X_test)
print predicted
print expected

from sklearn.cross_validation import KFold

def run_cv(X,y,clf_class,**kwargs):
    kf = KFold(len(y),n_folds=10,shuffle=True)
    y_pred = y.copy()

    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred

def accuracy(y_true,y_pred):
    return np.mean(y_true == y_pred)
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.naive_bayes import BernoulliNB as BNB
from sklearn.ensemble import GradientBoostingClassifier as GBC
print "RandomForestClassifier:"
print "%.3f" % accuracy(y_all, run_cv(X_all,y_all,RFC))
print "BernoulliNB:"
print "%.3f" % accuracy(y_all, run_cv(X_all,y_all,BNB))
print "GradientBoostingClassifier:"
print "%.3f" % accuracy(y_all, run_cv(X_all,y_all,GBC))