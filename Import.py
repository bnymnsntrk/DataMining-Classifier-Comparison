
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, precision_recall_fscore_support, roc_auc_score
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import time



data = pd.read_csv("data.csv")              #read data
data = shuffle(data)                        #shuffle data
data.reset_index(inplace=True, drop=True)   #reset indexes to make shuffle work

x = data[['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']]      #features
y = data['class']                                                                                                   #label

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)       #train


#~~~~~~~~~~ Naive Bayes ~~~~~~~~~~#
tic = time.perf_counter()
classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)*100
toc = time.perf_counter()
print(f"\nNaive Bayes done in {toc - tic:0.4f} seconds")
print("Accuracy for Naive Bayes     = ", "%0.2f" % accuracy)
#~~~~~~~~~~ Naive Bayes ~~~~~~~~~~#


#~~~~~~~~~~ Decision Tree ~~~~~~~~~~#
tic = time.perf_counter()
classifier2 = DecisionTreeClassifier()
classifier2.fit(x_train, y_train)

y_pred2 = classifier2.predict(x_test)
accuracy2 = accuracy_score(y_test, y_pred2)*100
toc = time.perf_counter()
print(f"\nDecision Tree done in {toc - tic:0.4f} seconds")
print("Accuracy for Decision Tree   = ", "%0.2f" % accuracy2)
#~~~~~~~~~~ Decision Tree ~~~~~~~~~~#


#~~~~~~~~~~ Support Vector Machine ~~~~~~~~~~#
tic = time.perf_counter()
classifier3 = svm.SVC(kernel='rbf')
classifier3.fit(x_train, y_train)

y_pred3 = classifier3.predict(x_test)
accuracy3 = accuracy_score(y_test, y_pred3)*100
toc = time.perf_counter()
print(f"\nSVM done in {toc - tic:0.4f} seconds")
print("Accuracy for SVM             = ", "%0.2f" % accuracy3)
#~~~~~~~~~~ Support Vector Machine ~~~~~~~~~~#