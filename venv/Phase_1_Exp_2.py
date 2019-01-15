print(__doc__)

import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt

from sklearn import datasets
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve
from sklearn import model_selection

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from termcolor import colored

features = pd.read_csv('Data/sop_and_stv.csv')
fullfeatures = pd.read_csv('Data/sop_stv_strength.csv')
testFeatures = pd.read_csv('Data/YAGO_sop_stv_values.csv')
fullTestFeatures = pd.read_csv('Data/YAGO_sop_stv_strength_values.csv')
print(fullTestFeatures.head(5))

print(colored('\nThe shape of our Test features is:','green'), testFeatures.shape)

print()
print(colored('\n     DESCRIPTIVE STATISTICS\n','yellow'))
print(colored(testFeatures.describe(),'cyan'))

features = pd.get_dummies(features)
fullfeatures = pd.get_dummies(fullfeatures)
testFeatures = pd.get_dummies(testFeatures)
fulltestFeatures = pd.get_dummies(fullTestFeatures)

features.iloc[:,5:].head(5)
fullfeatures.iloc[:,5:].head(5)
testFeatures.iloc[:,5].head(5)
fullTestFeatures.iloc[:,5].head(5)

labels = np.array(features['MturkTruth'])
fulllabels = np.array(fullfeatures['MturkTruth'])
testlabels = np.array(testFeatures['MturkTruth'])
fullTestlabels = np.array(fullTestFeatures['MturkTruth'])

count_0 = 0
count_1 = 0
for u in range(len(testlabels)):
    if(testlabels[u] == 0):
        count_0 += 1
    else:
        count_1 += 1

features= features.drop('MturkTruth', axis = 1)
fullfeatures= fullfeatures.drop('MturkTruth', axis = 1)
testFeatures = testFeatures.drop('MturkTruth', axis = 1)
fullTestFeatures = fullTestFeatures.drop('MturkTruth', axis = 1)

feature_list = list(features.columns)
fullfeature_list = list(fullfeatures.columns)
testFeature_list = list(testFeatures.columns)
fullTestFeature_list = list(fullTestFeatures.columns)

def add_missing_dummy_columns(d, columns):
    missing_cols = set(columns) - set(d.columns)
    for c in missing_cols:
        d[c] = 0


def fix_columns(d, columns):
    add_missing_dummy_columns(d, columns)

    # make sure we have all the columns we need
    assert (set(columns) - set(d.columns) == set())

    extra_cols = set(d.columns) - set(columns)
    if extra_cols: print("extra columns:", extra_cols)

    d = d[columns]
    return d


testFeatures = fix_columns(testFeatures, features.columns)
fullTestFeatures = fix_columns(fullTestFeatures, fullfeatures.columns)

features = np.array(features)
testFeatures = np.array(testFeatures)
fullTestFeatures = np.array(fullTestFeatures)

train_samples = 100

from sklearn.metrics import precision_recall_fscore_support

import pickle

print()
pred = np.empty([1416])

for i in range(len(pred)):
    pred[i] = 1

print(colored("Baseline",'green'))
print("Precision, Recall & F1 : (0.9265422013150755, 0.9625706214689266, 0.9442128514301166, None)")

loaded_model_RFC = pickle.load(open('other/SOPmodel_RFC', 'rb'))
pred1 = loaded_model_RFC.predict(testFeatures)
result_RFC = loaded_model_RFC.score(testFeatures, testlabels)

loaded_model_SVC = pickle.load(open('other/SOPmodel_SVC', 'rb'))
pred2 = loaded_model_SVC.predict(testFeatures)
result_SVC = loaded_model_SVC.score(testFeatures, testlabels)

loaded_model_GPC = pickle.load(open('other/SOPmodel_Gaussian', 'rb'))
pred3 = loaded_model_GPC.predict(testFeatures)
result_GPC = loaded_model_GPC.score(testFeatures, testlabels)


loaded_model_SGD = pickle.load(open('other/SOPmodel_SGD', 'rb'))
pred4 = loaded_model_SGD.predict(testFeatures)
result_SGD = loaded_model_SGD.score(testFeatures, testlabels)


print()

counto = 0
counti = 0
for i in range(len(testlabels)):
    if(pred1[i] == 1):
        counto += 1
    else:
        counti += 1

print(colored("Support Vector Classifier",'green'))

counto = 0
counti = 0
for i in range(len(testlabels)):
    if(pred2[i] == 1):
        counto += 1
    else:
        counti += 1

print("Precision, Recall & F1: ", precision_recall_fscore_support(testlabels, pred2, average='weighted'))
print()
counto = 0
counti = 0
for i in range(len(testlabels)):
    if(pred3[i] == 1):
        counto += 1
    else:
        counti += 1

print(colored("SGD Classifier",'green'))

counto = 0
counti = 0
for i in range(len(testlabels)):
    if(pred4[i] == 1):
        counto += 1
    else:
        counti += 1

print("Precision, Recall & F1: ", precision_recall_fscore_support(testlabels, pred4, average='weighted'))


print()
print(colored("Gaussian Process Classifier",'green'))

counto = 0
counti = 0
for i in range(len(fullTestlabels)):
    if(pred3[i] == 1):
        counto += 1
    else:
        counti += 1

print("Precision, Recall & F1: ", precision_recall_fscore_support(fullTestlabels, pred3, average='weighted'))
print()


print(colored("Random Forest Classifier",'green'))

counto = 0
counti = 0
for i in range(len(fullTestlabels)):
    if(pred4[i] == 1):
        counto += 1
    else:
        counti += 1
print("Precision, Recall & F1: ", precision_recall_fscore_support(fullTestlabels, pred1, average='weighted'))