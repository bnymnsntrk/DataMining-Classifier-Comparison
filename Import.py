
from IPython.display import display
from numpy.random import RandomState
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB


data = pd.read_csv("data.csv")

features = np.array(pd.DataFrame(data, columns=['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']))
label = np.array(pd.DataFrame(data, columns=['class']))


model = GaussianNB()
model.fit(features, label.ravel())

predicted= model.predict([[48.2468, 17.3565, 3.0332, 0.2529, 0.1515, 8.573, 38.0957, 10.5868,4.792, 219.087]])
print("Predicted Value:", predicted)
