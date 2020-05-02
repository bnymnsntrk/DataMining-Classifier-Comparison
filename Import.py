
from IPython.display import display
from numpy.random import RandomState
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
import random
import operator

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.utils import shuffle


data = pd.read_csv("data.csv")              #read data
data = shuffle(data)                        #shuffle data
data.reset_index(inplace=True, drop=True)   #reset indexes to make shuffle work
training_length = len(data)*0.7             #%70 of data is training data

training_data = data.loc[:training_length, :]      #first %70 of data goes to training data
test_data = data.loc[training_length+1:, :]        #by adding 1 to index, we pick remaning %30 to test data

print(data)



#features = np.array(pd.DataFrame(data, columns=['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']))
#label = np.array(pd.DataFrame(data, columns=['class']))


#model.fit(features, label.ravel())
#predicted = model.predict([[48.2468, 17.3565, 3.0332, 0.2529, 0.1515, 8.573, 38.0957, 10.5868,4.792, 219.087]])
#print("Predicted Value:", predicted)
