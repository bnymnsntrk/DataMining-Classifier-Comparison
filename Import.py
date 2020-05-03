
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)       #train and test split

#print(data)

#~~~~~~~~~~ Naive Bayes ~~~~~~~~~~#
timer1 = time.perf_counter()                     #timer starts
classifier = GaussianNB()                        #create classifier
classifier.fit(x_train, y_train)                 #train the model

y_pred = classifier.predict(x_test)              #predict the value
accuracy = accuracy_score(y_test, y_pred)*100    #calculate accuracy
timer2 = time.perf_counter()                     #timer ends
print(f"\nNaive Bayes done in {timer2 - timer1:0.4f} seconds")
print("Accuracy for Naive Bayes     = ", "%0.2f" % accuracy)
print(classification_report(y_test,y_pred))
#~~~~~~~~~~ Naive Bayes ~~~~~~~~~~#


#~~~~~~~~~~ Decision Tree ~~~~~~~~~~#
timer1 = time.perf_counter()
classifier2 = DecisionTreeClassifier()
classifier2.fit(x_train, y_train)

y_pred2 = classifier2.predict(x_test)
accuracy2 = accuracy_score(y_test, y_pred2)*100
timer2 = time.perf_counter()
print(f"\nDecision Tree done in {timer2 - timer1:0.4f} seconds")
print("Accuracy for Decision Tree   = ", "%0.2f" % accuracy2)
print(classification_report(y_test,y_pred2))
#~~~~~~~~~~ Decision Tree ~~~~~~~~~~#


#~~~~~~~~~~ Support Vector Machine ~~~~~~~~~~#
timer1 = time.perf_counter()
classifier3 = svm.SVC(kernel='rbf')
classifier3.fit(x_train, y_train)

y_pred3 = classifier3.predict(x_test)
accuracy3 = accuracy_score(y_test, y_pred3)*100
timer2 = time.perf_counter()
print(f"\nSVM done in {timer2 - timer1:0.4f} seconds")
print("Accuracy for SVM             = ", "%0.2f" % accuracy3)
print(classification_report(y_test,y_pred3))
#~~~~~~~~~~ Support Vector Machine ~~~~~~~~~~#