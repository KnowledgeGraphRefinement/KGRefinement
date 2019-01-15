print(__doc__)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt

from sklearn import datasets
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve
from sklearn import model_selection


import pandas as pd
import numpy as np
from termcolor import colored

features = pd.read_csv('Data/sop_and_stv.csv')
print(features.head(5))

print(colored('\nThe shape of our features is:','green'), features.shape)

print()
print(colored('\n     DESCRIPTIVE STATISTICS\n','yellow'))
print(colored(features.describe(),'cyan'))

features = pd.get_dummies(features)

features.iloc[:,5:].head(5)

labels = np.array(features['MturkTruth'])

features= features.drop('MturkTruth', axis = 1)

feature_list = list(features.columns)


def add_missing_dummy_columns(d, columns):
    missing_cols = set(columns) - set(d.columns)
    for c in missing_cols:
        d[c] = 0


def fix_columns(d, columns):
    add_missing_dummy_columns(d, columns)

    assert (set(columns) - set(d.columns) == set())

    extra_cols = set(d.columns) - set(columns)
    if extra_cols: print("extra columns:", extra_cols)

    d = d[columns]
    return d


testFeatures = fix_columns(features, features.columns)

testFeatures = np.array(testFeatures)

train_samples = 100

X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size = 0.25, random_state = 42)

print(colored('\n    TRAINING & TESTING SETS','yellow'))
print(colored('\nTraining Features Shape:','magenta'), X_train.shape)
print(colored('Training Labels Shape:','magenta'), X_test.shape)
print(colored('Testing Features Shape:','magenta'), y_train.shape)
print(colored('Testing Labels Shape:','magenta'), y_test.shape)

print()


lr = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
gnb = GaussianProcessClassifier(random_state=0, kernel=Matern(1.0))
svc = LinearSVC(C=1.0)
rfc = RandomForestClassifier(n_estimators=100)

# Classifier calibration
from sklearn.metrics import precision_recall_fscore_support

m_y_test = np.empty([465])

for x in range(len(m_y_test)):
    m_y_test[x] = 1
print(colored("Baseline",'green'))
print("Precision, Recall & F1: ", precision_recall_fscore_support(y_test, m_y_test, average='weighted'))
print()
plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for clf, name in [(lr, 'SGD Classifier'),
                  (gnb, 'Gaussian Process Classifier'),
                  (svc, 'Support Vector Classifier'),
                  (rfc, 'Random Forest Classifier')]:
    clf.fit(X_train, y_train)
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict(X_test)
    else:  # use decision function
        prob_pos = clf.decision_function(X_test)
        prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        prob_pos = np.round(prob_pos,0)
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_test, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (name, ))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,histtype="step", lw=2)

    print(colored(name,'green'))
    print("Precision, Recall & F1: ", precision_recall_fscore_support(y_test, prob_pos, average='weighted'))
    print()
