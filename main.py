import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.cross_validation import KFold
import plot


def swap(index1, index2, iterable):
    for x in iterable:
        x[index1], x[index2] = x[index2], x[index1]


def download_matrix(path):
    # TODO: don't understand first argument in matrix
    raw_data = open(path)
    data_set = np.loadtxt(raw_data, delimiter=",")
    # TODO: unused step (?) swap o and 6 column
    matrix_from_file = data_set[:, :]
    swap(0, 6, matrix_from_file)
    # print 'matrix', matrix_from_file
    # print 'data', data_set
    return matrix_from_file, data_set


def prepare_matrix_for_feature_engineering(matrix_from_file, data_set):
    x_all = matrix_from_file[:, 1:]
    # + matrix_from_file[:, 7:13] + matrix_from_file[:, 13:19] + matrix_from_file[:, 19:25]
    # print "x_all", x_all
    y_all = data_set[:, 0]
    # print y_all
    x_train = matrix_from_file[0:150, 1:]
    # matrix_from_file[0:150,0:6] + matrix_from_file[0:150, 7:13] + matrix_from_file[0:150, 13:19] +
    # matrix_from_file[0:150, 19:25]
    # print 'x_train', x_train
    y_train = data_set[0:150, 0]
    # print 'y_train', y_train
    x_test = data_set[150:194, 1:]
    # data_set[150:194,0:6] + matrix_from_file[150:194, 7:13] + matrix_from_file[150:194, 13:19] +
    #  matrix_from_file[150:194, 19:25]
    y_test = data_set[150:194, 0]
    # data_set[150:194,6]
    # print 'x_test', x_test
    # print 'y_test', y_test
    return x_train, y_train, x_test, y_test, x_all, y_all


def gaussian_nb(x_train, y_train, x_test, y_test):
    model = gnb()
    model.fit(x_train, y_train)
    expected = y_test
    predicted = model.predict(x_test)
    # print expected
    # print predicted
    return expected, predicted


def random_forest_classifier(x_train, y_train, x_test, y_test):
    model = rfc(n_estimators=100)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    expected = y_test
    # print expected
    # print predicted
    return expected, predicted


def gradient_boosting_classifier(x_train, y_train, x_test, y_test):
    model = gbc(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2,
                min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=1, init=None, random_state=None,
                max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)
    model.fit(x_train, y_train)
    expected = y_test
    predicted = model.predict(x_test)
    # print expected
    # print predicted
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
    _matrix_from_file, _data_set = download_matrix("./data.txt")
    _x_train, _y_train, _x_test, _y_test, _x_all, _y_all = prepare_matrix_for_feature_engineering(_matrix_from_file,
                                                                                                  _data_set)
    # TODO: why we don't use normalization?
    # normalize the data attributes
    normalized_x = preprocessing.normalize(_x_all)
    # print "normalized_x", normalized_x
    # standardize the data attributes
    standardized_x = preprocessing.scale(_x_all)
    # print "standardized_x", standardized_x

    scaler = StandardScaler()
    _x_all = scaler.fit_transform(_x_all)
    # print "_x_all", _x_all

    # TODO: where use expected, predicted results?
    # TODO: may use for loop for getting stable results
    expected_gnb, predicted_gnb = gaussian_nb(_x_train, _y_train, _x_test, _y_test)
    expected_rfc, predicted_rfc = random_forest_classifier(_x_train, _y_train, _x_test, _y_test)
    expected_gbc, predicted_gbc = gradient_boosting_classifier(_x_train, _y_train, _x_test, _y_test)
    # TODO: create graphics on base confusion matrix (see doc)
    # print(expected_gnb, predicted_gnb)
    confusion_matrix_gnb = metrics.confusion_matrix(expected_gnb, predicted_gnb)
    # print confusion_matrix_gnb
    plot.plot_classification_report(confusion_matrix_gnb)
    # print(expected_rfc, predicted_rfc)
    confusion_matrix_frc = metrics.confusion_matrix(expected_rfc, predicted_rfc)
    plot.plot_classification_report(confusion_matrix_frc)
    # print(expected_gbc, predicted_gbc)
    confusion_matrix_gbc = metrics.confusion_matrix(expected_gbc, predicted_gbc)
    plot.plot_classification_report(confusion_matrix_gbc)

    #TODO: plot classification report
    classification_report = metrics.classification_report(expected_gnb, predicted_gnb)
    print classification_report



    print "RandomForestClassifier:"
    print "%.3f" % accuracy(_y_all, run_cross_validation(_x_all, _y_all, rfc))
    print "%.3f" % accuracy(_y_all, run_cross_validation(normalized_x, _y_all, rfc))
    print "%.3f" % accuracy(_y_all, run_cross_validation(standardized_x, _y_all, rfc))
    print "GaussianNB:"
    print "%.3f" % accuracy(_y_all, run_cross_validation(_x_all, _y_all, gnb))
    print "%.3f" % accuracy(_y_all, run_cross_validation(normalized_x, _y_all, gnb))
    print "%.3f" % accuracy(_y_all, run_cross_validation(standardized_x, _y_all, gnb))
    print "GradientBoostingClassifier:"
    print "%.3f" % accuracy(_y_all, run_cross_validation(_x_all, _y_all, gbc))
    print "%.3f" % accuracy(_y_all, run_cross_validation(normalized_x, _y_all, gbc))
    print "%.3f" % accuracy(_y_all, run_cross_validation(standardized_x, _y_all, gbc))
