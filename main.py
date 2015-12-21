import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.naive_bayes import BernoulliNB as bnb
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn.cross_validation import KFold
import plot


def swap(index1, index2, iterable):
    for x in iterable:
        x[index1], x[index2] = x[index2], x[index1]


def download_matrix(path):
    raw_data = open(path)
    data_set = np.loadtxt(raw_data, delimiter=",")
    matrix_from_file = data_set[:, :]
    swap(0, 6, matrix_from_file)
    return matrix_from_file


def prepare_matrix_for_feature_engineering(matrix_from_file):
    x_all = matrix_from_file[:, 1:]
    y_all = matrix_from_file[:, 0]
    x_train = matrix_from_file[0:150, 1:]
    y_train = matrix_from_file[0:150, 0]
    x_test = matrix_from_file[150:194, 1:]
    y_test = matrix_from_file[150:194, 0]
    return x_train, y_train, x_test, y_test, x_all, y_all

# from sklearn.naive_bayes import GaussianNB as gnb
# def gaussian_nb(x_train, y_train, x_test, y_test):
#     model = gnb()
#     model.fit(x_train, y_train)
#     expected = y_test
#     predicted = model.predict(x_test)
#     return expected, predicted


def naive_bayes_bnb(x_train, y_train, x_test, y_test):
    model = bnb()
    model.fit(x_train, y_train)
    expected = y_test
    predicted = model.predict(x_test)
    return expected, predicted

def naive_bayes_mnb(x_train, y_train, x_test, y_test):
    model = mnb()
    model.fit(x_train, y_train)
    expected = y_test
    predicted = model.predict(x_test)
    return expected, predicted

def random_forest_classifier(x_train, y_train, x_test, y_test):
    model = rfc(n_estimators=1000, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                min_weight_fraction_leaf=0.0, max_features="auto", max_leaf_nodes=None, bootstrap=False,
                oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    expected = y_test
    return expected, predicted


# def random_forest_classifier(x_train, y_train, x_test, y_test):
#     model = rfc(n_estimators=100)
#     model.fit(x_train, y_train)
#     predicted = model.predict(x_test)
#     expected = y_test
#     # print expected
#     # print predicted
#     return expected, predicted


def gradient_boosting_classifier(x_train, y_train, x_test, y_test):
    model = gbc(loss='deviance', learning_rate=0.2, n_estimators=10, subsample=1.0, min_samples_split=2,
                min_samples_leaf=10, min_weight_fraction_leaf=0.0, max_depth=5, init=None, random_state=None,
                max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)
    model.fit(x_train, y_train)
    expected = y_test
    predicted = model.predict(x_test)
    return expected, predicted


def run_cross_validation(x, y, clf_class, **kwargs):
    kf = KFold(len(y), n_folds=10, shuffle=True)
    y_prediction = y.copy()

    for train_index, test_index in kf:
        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(x_train, y_train)
        y_prediction[test_index] = clf.predict(x_test)
    return y_prediction


def accuracy(y_true, y_prediction):
    return np.mean(y_true == y_prediction)


if __name__ == "__main__":
    _matrix_from_file = download_matrix("./data.txt")
    _x_train, _y_train, _x_test, _y_test, _x_all, _y_all = prepare_matrix_for_feature_engineering(_matrix_from_file)

    scaler = StandardScaler()
    _x_all = scaler.fit_transform(_x_all)

    expected_bnb, predicted_bnb = naive_bayes_bnb(_x_train, _y_train, _x_test, _y_test)
    expected_mnb, predicted_mnb = naive_bayes_mnb(_x_train, _y_train, _x_test, _y_test)
    expected_rfc, predicted_rfc = random_forest_classifier(_x_train, _y_train, _x_test, _y_test)
    expected_gbc, predicted_gbc = gradient_boosting_classifier(_x_train, _y_train, _x_test, _y_test)

    # confusion_matrix_gnb = metrics.confusion_matrix(expected_bnb, predicted_bnb)
    # plot.plot_classification_report(confusion_matrix_gnb)
    # confusion_matrix_frc = metrics.confusion_matrix(expected_rfc, predicted_rfc)
    # plot.plot_classification_report(confusion_matrix_frc)
    # confusion_matrix_gbc = metrics.confusion_matrix(expected_gbc, predicted_gbc)
    # plot.plot_classification_report(confusion_matrix_gbc)


    # TODO: plot classification report
    # classification_report = metrics.classification_report(expected_gnb, predicted_gnb)
    # print classification_report

    print "RandomForestClassifier:"
    print "%.3f" % accuracy(_y_all, run_cross_validation(_x_all, _y_all, rfc))
    print "BernoulliNB:"
    print "%.3f" % accuracy(_y_all, run_cross_validation(_x_all, _y_all, bnb))
    print "GradientBoostingClassifier:"
    print "%.3f" % accuracy(_y_all, run_cross_validation(_x_all, _y_all, gbc))

    print "bnb"
    classification_report = metrics.classification_report(expected_bnb, predicted_bnb)
    print classification_report
    plot.plot_classification_report_for_each_method(classification_report,'Classification report for BernoulliNB')

    print "mnb"
    classification_report = metrics.classification_report(expected_mnb, predicted_mnb)
    print classification_report
    plot.plot_classification_report_for_each_method(classification_report,'Classification report for MultinomialNB')

    print "rfc"
    classification_report = metrics.classification_report(expected_rfc, predicted_rfc)
    print classification_report
    plot.plot_classification_report_for_each_method(classification_report,'Classification report for RandomForestClassifier')


    print "gbc"
    classification_report = metrics.classification_report(expected_gbc, predicted_gbc)
    print classification_report
    plot.plot_classification_report_for_each_method(classification_report,'Classification report for GradientBoostingClassifier')



    print "RandomForestClassifier:"
    print "%.3f" % accuracy(_y_all, run_cross_validation(_x_all, _y_all, rfc))
    # print "%.3f" % accuracy(_y_all, run_cross_validation(normalized_x, _y_all, rfc))
    # print "%.3f" % accuracy(_y_all, run_cross_validation(standardized_x, _y_all, rfc))
    print "GaussianNB:"
    print "%.3f" % accuracy(_y_all, run_cross_validation(_x_all, _y_all, bnb))
    # print "%.3f" % accuracy(_y_all, run_cross_validation(normalized_x, _y_all, gnb))
    # print "%.3f" % accuracy(_y_all, run_cross_validation(standardized_x, _y_all, gnb))
    print "GradientBoostingClassifier:"
    print "%.3f" % accuracy(_y_all, run_cross_validation(_x_all, _y_all, gbc))
    # print "%.3f" % accuracy(_y_all, run_cross_validation(normalized_x, _y_all, gbc))
    # print "%.3f" % accuracy(_y_all, run_cross_validation(standardized_x, _y_all, gbc))

