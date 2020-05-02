
from IPython.display import display
from numpy.random import RandomState
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, precision_recall_fscore_support, roc_auc_score
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split


data = pd.read_csv("data.csv")              #read data
data = shuffle(data)                        #shuffle data
data.reset_index(inplace=True, drop=True)   #reset indexes to make shuffle work
training_length = len(data)*0.7             #%70 of data is training data

x = data[['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']]      #features
y = data['class']                                                                                                   #label

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)       #train

classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)*100
print(accuracy)

#features = np.array(pd.DataFrame(data, columns=['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']))
#label = np.array(pd.DataFrame(data, columns=['class']))


#model.fit(features, label.ravel())
#predicted = model.predict([[48.2468, 17.3565, 3.0332, 0.2529, 0.1515, 8.573, 38.0957, 10.5868,4.792, 219.087]])
#print("Predicted Value:", predicted)
